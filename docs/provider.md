# Groq

Groq provider for PapiAI. Ultra-fast inference powered by Groq LPU (Language Processing Unit) hardware.

## Installation

```bash
composer require papi-ai/groq
```

## Usage

```php
use PapiAI\Core\Agent;
use PapiAI\Groq\GroqProvider;

$provider = new GroqProvider(
    apiKey: $_ENV['GROQ_API_KEY'],
);

$agent = new Agent(
    provider: $provider,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Models

```php
GroqProvider::MODEL_GPT_OSS_120B  // 'openai/gpt-oss-120b' (default)
GroqProvider::MODEL_GPT_OSS_20B   // 'openai/gpt-oss-20b' (fast)
```

The Llama and Mixtral constants are still shipped but deprecated. Mixtral was decommissioned on 20 March 2025, and both Llama models are decommissioned on **16 August 2026**.


## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Vision | Yes |
| Structured output | Yes |

Groq uses custom LPU (Language Processing Unit) hardware for ultra-fast inference. It is designed for low-latency, high-throughput workloads and can deliver significantly faster token generation compared to traditional GPU-based providers.

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
