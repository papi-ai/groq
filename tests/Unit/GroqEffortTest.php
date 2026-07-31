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

use PapiAI\Core\Effort;
use PapiAI\Core\Message;
use PapiAI\Groq\GroqProvider;

/**
 * Captures the request payload so effort mapping can be asserted without HTTP.
 */
class TestableGroqEffortProvider extends GroqProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['choices' => [['message' => ['role' => 'assistant', 'content' => 'ok'], 'finish_reason' => 'stop']]];
    }
}

describe('GroqProvider reasoning effort', function () {
    beforeEach(function () {
        $this->provider = new TestableGroqEffortProvider('test-api-key');
        $this->chat = fn (array $options) => $this->provider->chat([Message::user('hi')], $options);
    });

    it('maps the three levels gpt-oss accepts', function () {
        foreach (['low', 'medium', 'high'] as $level) {
            ($this->chat)(['effort' => $level]);

            expect($this->provider->lastPayload['reasoning_effort'])->toBe($level);
        }
    });

    it('narrows the ends of the scale onto that range', function () {
        ($this->chat)(['effort' => 'none']);
        expect($this->provider->lastPayload['reasoning_effort'])->toBe('low');

        ($this->chat)(['effort' => 'maximum']);
        expect($this->provider->lastPayload['reasoning_effort'])->toBe('high');
    });

    it('stays quiet on models that are not gpt-oss', function () {
        // Groq hosts third-party models whose reasoning controls differ or do not exist.
        ($this->chat)(['effort' => 'high', 'model' => 'qwen/qwen3.6-27b']);

        expect($this->provider->lastPayload)->not->toHaveKey('reasoning_effort');
    });

    it('sends nothing when the caller does not ask', function () {
        ($this->chat)([]);

        expect($this->provider->lastPayload)->not->toHaveKey('reasoning_effort');
    });

    it('rejects a level it does not recognise', function () {
        expect(fn () => ($this->chat)(['effort' => 'enormous']))
            ->toThrow(InvalidArgumentException::class, 'enormous');
    });

    it('accepts a provider-level default the call can override', function () {
        $provider = new TestableGroqEffortProvider('k', 'openai/gpt-oss-120b', 4096, Effort::High);

        $provider->chat([Message::user('hi')], []);
        expect($provider->lastPayload['reasoning_effort'])->toBe('high');

        $provider->chat([Message::user('hi')], ['effort' => 'low']);
        expect($provider->lastPayload['reasoning_effort'])->toBe('low');
    });
});
