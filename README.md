# PapiAI Groq Provider

[![Tests](https://github.com/papi-ai/groq/workflows/CI/badge.svg)](https://github.com/papi-ai/groq/actions?query=workflow%3ACI)

Groq provider for [PapiAI](https://github.com/papi-ai/papi-core) - A simple but powerful PHP library for building AI agents.

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

## Available Models

```php
GroqProvider::MODEL_LLAMA_3_3_70B  // 'llama-3.3-70b-versatile' (default)
GroqProvider::MODEL_LLAMA_3_1_8B   // 'llama-3.1-8b-instant' (fast)
GroqProvider::MODEL_MIXTRAL_8X7B   // 'mixtral-8x7b-32768'
```

## Features

- Ultra-fast inference via Groq LPU
- Tool/function calling
- Streaming support

## License

MIT
