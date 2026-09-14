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

use PapiAI\Groq\GroqModel;
use PapiAI\Groq\GroqProvider;

describe('GroqModel', function () {
    it('is the source of truth the old constants alias', function () {
        expect(GroqProvider::MODEL_GPT_OSS_120B)->toBe(GroqModel::GptOss120b->value);
    });

    it('ships unique IDs', function () {
        $ids = array_map(fn (GroqModel $m) => $m->value, GroqModel::cases());

        expect($ids)->toBe(array_unique($ids));
    });

    it('returns null for an ID it has not heard of, rather than throwing', function () {
        expect(GroqModel::tryFrom('not-a-model'))->toBeNull();
    });

    it('knows which models are retired, when, and what replaces them', function () {
        expect(GroqModel::Llama33Versatile->isDeprecated())->toBeTrue();
        expect(GroqModel::Llama33Versatile->retiredOn())->toBe('2026-08-16');
        expect(GroqModel::Llama33Versatile->replacement())->toBe(GroqModel::GptOss120b);
        expect(GroqModel::GptOss120b->isDeprecated())->toBeFalse();
    });
});
