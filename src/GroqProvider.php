<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

namespace PapiAI\Groq;

use Generator;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use RuntimeException;

/**
 * Groq API provider for PapiAI.
 *
 * Bridges PapiAI's core types with the Groq Cloud API (OpenAI-compatible format),
 * handling format conversion. Powered by Groq's LPU (Language Processing Unit)
 * inference hardware for ultra-low-latency responses. Supports chat completions,
 * streaming, tool use, vision, and structured output. Authentication via Bearer token.
 * All HTTP via ext-curl.
 *
 * Supported models include:
 * - llama-3.3-70b-versatile (general purpose)
 * - llama-3.1-8b-instant (fast inference)
 * - mixtral-8x7b-32768 (Mixtral)
 *
 * @see https://console.groq.com/docs/api-reference
 */
class GroqProvider implements ProviderInterface
{
    private const API_URL = 'https://api.groq.com/openai/v1/chat/completions';

    public const MODEL_LLAMA_3_3_70B = 'llama-3.3-70b-versatile';
    public const MODEL_LLAMA_3_1_8B = 'llama-3.1-8b-instant';
    public const MODEL_MIXTRAL_8X7B = 'mixtral-8x7b-32768';

    /**
     * Create a new Groq provider instance.
     *
     * @param string $apiKey       Groq API key for Bearer token authentication
     * @param string $defaultModel Default model identifier for chat requests
     * @param int    $defaultMaxTokens Default maximum tokens for responses
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_LLAMA_3_3_70B,
        private readonly int $defaultMaxTokens = 4096,
    ) {
    }

    /**
     * Send a chat completion request to the Groq API.
     *
     * @param Message[] $messages Conversation messages to send
     * @param array     $options  Options: model, maxTokens, temperature, stopSequences, outputSchema, tools
     *
     * @return Response The parsed API response
     *
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When the rate limit is exceeded
     * @throws ProviderException       When the API returns an error
     */
    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromOpenAI($response, $messages);
    }

    /**
     * Stream a chat completion response from the Groq API.
     *
     * Yields StreamChunk objects as server-sent events are received.
     *
     * @param Message[] $messages Conversation messages to send
     * @param array     $options  Options: model, maxTokens, temperature, stopSequences, outputSchema, tools
     *
     * @return iterable<StreamChunk> Stream of response chunks
     */
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($payload) as $event) {
            $delta = $event['choices'][0]['delta'] ?? [];
            if (isset($delta['content'])) {
                yield new StreamChunk($delta['content']);
            }
            if (($event['choices'][0]['finish_reason'] ?? null) !== null) {
                yield new StreamChunk('', isComplete: true);
            }
        }
    }

    /**
     * Indicates that Groq supports tool/function calling.
     */
    public function supportsTool(): bool
    {
        return true;
    }

    /**
     * Indicates that Groq supports vision/image inputs.
     */
    public function supportsVision(): bool
    {
        return true;
    }

    /**
     * Indicates that Groq supports structured JSON output via json_schema.
     */
    public function supportsStructuredOutput(): bool
    {
        return true;
    }

    /**
     * Return the provider identifier.
     */
    public function getName(): string
    {
        return 'groq';
    }

    /**
     * Build the API request payload.
     */
    private function buildPayload(array $messages, array $options): array
    {
        $apiMessages = [];

        foreach ($messages as $message) {
            if ($message instanceof Message) {
                $apiMessages[] = $this->convertMessage($message);
            }
        }

        $payload = [
            'model' => $options['model'] ?? $this->defaultModel,
            'messages' => $apiMessages,
        ];

        if (isset($options['maxTokens'])) {
            $payload['max_tokens'] = $options['maxTokens'];
        }

        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $payload['stop'] = $options['stopSequences'];
        }

        // Handle structured output / JSON mode
        if (isset($options['outputSchema'])) {
            $payload['response_format'] = [
                'type' => 'json_schema',
                'json_schema' => [
                    'name' => 'response',
                    'schema' => $options['outputSchema'],
                ],
            ];
        }

        // Handle tools
        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = $this->convertTools($options['tools']);
        }

        return $payload;
    }

    /**
     * Convert a Message to OpenAI-compatible API format.
     */
    private function convertMessage(Message $message): array
    {
        $apiMessage = [
            'role' => $this->convertRole($message->role),
        ];

        if ($message->isTool()) {
            $apiMessage['role'] = 'tool';
            $apiMessage['content'] = $message->content;
            $apiMessage['tool_call_id'] = $message->toolCallId;
        } elseif ($message->hasToolCalls()) {
            $apiMessage['content'] = $message->getText() ?: null;
            $apiMessage['tool_calls'] = array_map(function (ToolCall $tc) {
                return [
                    'id' => $tc->id,
                    'type' => 'function',
                    'function' => [
                        'name' => $tc->name,
                        'arguments' => json_encode($tc->arguments),
                    ],
                ];
            }, $message->toolCalls);
        } elseif (is_array($message->content)) {
            $apiMessage['content'] = $this->convertMultimodalContent($message->content);
        } else {
            $apiMessage['content'] = $message->content;
        }

        return $apiMessage;
    }

    /**
     * Convert multimodal content to OpenAI-compatible format.
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['type' => 'text', 'text' => $part['text']];
            } elseif ($part['type'] === 'image') {
                $source = $part['source'];
                if ($source['type'] === 'url') {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => ['url' => $source['url']],
                    ];
                } else {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => [
                            'url' => "data:{$source['media_type']};base64,{$source['data']}",
                        ],
                    ];
                }
            }
        }

        return $parts;
    }

    /**
     * Convert tools from PapiAI format to OpenAI-compatible format.
     */
    private function convertTools(array $tools): array
    {
        $openaiTools = [];

        foreach ($tools as $tool) {
            if (is_array($tool)) {
                $openaiTools[] = [
                    'type' => 'function',
                    'function' => [
                        'name' => $tool['name'],
                        'description' => $tool['description'],
                        'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                    ],
                ];
            }
        }

        return $openaiTools;
    }

    /**
     * Convert Role to OpenAI-compatible role string.
     */
    private function convertRole(Role $role): string
    {
        return match ($role) {
            Role::System => 'system',
            Role::User => 'user',
            Role::Assistant => 'assistant',
            Role::Tool => 'tool',
        };
    }

    /**
     * Send a synchronous POST request to the Groq API.
     *
     * @param array $payload The JSON-serializable request body
     *
     * @return array The decoded JSON response
     *
     * @throws RuntimeException        When the cURL request fails
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When the rate limit is exceeded
     * @throws ProviderException       When the API returns an error
     */
    protected function request(array $payload): array
    {
        $ch = curl_init(self::API_URL);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("Groq API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $this->throwForStatusCode($httpCode, $data);
        }

        return $data;
    }

    /**
     * Map HTTP error status codes to the appropriate PapiAI exception.
     *
     * @param int        $httpCode The HTTP response status code
     * @param array|null $data     The decoded error response body
     *
     * @throws AuthenticationException For 401 responses
     * @throws RateLimitException      For 429 responses
     * @throws ProviderException       For all other error responses
     */
    protected function throwForStatusCode(int $httpCode, ?array $data): never
    {
        $errorMessage = $data['error']['message'] ?? 'Unknown error';

        if ($httpCode === 401) {
            throw new AuthenticationException(
                $this->getName(),
                $httpCode,
                $data,
            );
        }

        if ($httpCode === 429) {
            throw new RateLimitException(
                $this->getName(),
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        throw new ProviderException(
            "Groq API error ({$httpCode}): {$errorMessage}",
            $this->getName(),
            $httpCode,
            $data,
        );
    }

    /**
     * Send a streaming POST request and yield parsed SSE events.
     *
     * Buffers the full response then parses server-sent events line by line.
     *
     * @param array $payload The JSON-serializable request body with stream=true
     *
     * @return Generator<int, array> Yields decoded JSON event data from each SSE data line
     */
    protected function streamRequest(array $payload): Generator
    {
        $ch = curl_init(self::API_URL);

        $buffer = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);
        curl_close($ch);

        // Parse SSE events
        $lines = explode("\n", $buffer);
        foreach ($lines as $line) {
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $json = substr($line, 6);
                if ($json === '[DONE]') {
                    break;
                }
                $event = json_decode($json, true);
                if ($event !== null) {
                    yield $event;
                }
            }
        }
    }
}
