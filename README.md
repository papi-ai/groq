# PapiAI Groq Provider

[![CI](https://github.com/papi-ai/groq/workflows/CI/badge.svg)](https://github.com/papi-ai/groq/actions?query=workflow%3ACI) [![Latest Version](https://img.shields.io/packagist/v/papi-ai/groq.svg)](https://packagist.org/packages/papi-ai/groq) [![Total Downloads](https://img.shields.io/packagist/dt/papi-ai/groq.svg)](https://packagist.org/packages/papi-ai/groq) [![PHP Version](https://img.shields.io/packagist/php-v/papi-ai/groq.svg)](https://packagist.org/packages/papi-ai/groq) [![License](https://img.shields.io/packagist/l/papi-ai/groq.svg)](https://packagist.org/packages/papi-ai/groq)

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
GroqProvider::MODEL_GPT_OSS_120B  // 'openai/gpt-oss-120b' (default)
GroqProvider::MODEL_GPT_OSS_20B   // 'openai/gpt-oss-20b' (fast)
```

The Llama and Mixtral constants are still shipped but deprecated. Mixtral was decommissioned on 20 March 2025, and both Llama models are decommissioned on **16 August 2026**.


## Features

- Ultra-fast inference via Groq LPU
- Tool/function calling
- Streaming support

## Creating It by Name

An application that picks its provider from configuration can build this one without naming
its class:

```php
use PapiAI\Core\Factory\ComposerDiscovery;
use PapiAI\Core\Factory\ProviderRegistry;

$registry = new ProviderRegistry(new ComposerDiscovery());

$provider = $registry->create('groq', [
    'api_key' => $_ENV['GROQ_API_KEY'],
]);
```

It takes `api_key` (required), `model`, `max_tokens` and `effort`. Anything it is not given falls
back to the constructor's defaults, and a missing or unknown setting throws `ConfigurationException`
before any request is made.

## License

MIT
