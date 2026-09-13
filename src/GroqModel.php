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

/**
 * Every Groq model this package knows.
 *
 * The enum is the source of truth: the `MODEL_*` constants on GroqProvider alias its values, so both
 * spell the same string. It lets a watchdog enumerate what we ship instead of parsing source, and
 * each case knows whether it has been retired and what replaces it.
 *
 * An ID we have not heard of is not an error: `tryFrom()` returns null and callers may pass it
 * straight through, since next month's model is far likelier than last year's.
 *
 * @see https://console.groq.com/docs/models
 */
enum GroqModel: string
{
    case GptOss120b = 'openai/gpt-oss-120b';
    case GptOss20b = 'openai/gpt-oss-20b';
    /** @deprecated Decommissioned 16 August 2026 for free and developer tiers; enterprise only since. Use GptOss120b. */
    case Llama33Versatile = 'llama-3.3-70b-versatile';
    /** @deprecated Decommissioned 16 August 2026 for free and developer tiers; enterprise only since. Use GptOss20b. */
    case Llama31Instant = 'llama-3.1-8b-instant';
    /** @deprecated Decommissioned 20 March 2025; requests fail. */
    case Mixtral8x7b = 'mixtral-8x7b-32768';

    /**
     * Whether the provider has retired this model.
     */
    public function isDeprecated(): bool
    {
        return match ($this) {
            self::Llama33Versatile => true,
            self::Llama31Instant => true,
            self::Mixtral8x7b => true,
            default => false,
        };
    }

    /**
     * The published retirement date, ISO formatted, where the provider gave one.
     */
    public function retiredOn(): ?string
    {
        return match ($this) {
            self::Llama33Versatile => '2026-08-16',
            self::Llama31Instant => '2026-08-16',
            self::Mixtral8x7b => '2025-03-20',
            default => null,
        };
    }

    /**
     * What to use instead, for retired models that have a successor here.
     */
    public function replacement(): ?self
    {
        return match ($this) {
            self::Llama33Versatile => self::GptOss120b,
            self::Llama31Instant => self::GptOss20b,
            default => null,
        };
    }
}
