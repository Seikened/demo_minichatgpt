import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';

const index = fs.readFileSync('src/demo_mini_chat/static/index.html', 'utf8');
const app = fs.readFileSync('src/demo_mini_chat/static/app.js', 'utf8');
const web = fs.readFileSync('src/demo_mini_chat/web.py', 'utf8');

test('presentation branding is precise and removes Mini ChatGPT', () => {
  assert.ok(index.includes('Inteligencia artificial generativa en procesamiento de lenguaje natural.'));
  assert.ok(!index.includes('Mini ChatGPT'));
  assert.ok(!web.includes('Mini ChatGPT'));
});

test('presentation removes duplicated token strip and concept card', () => {
  assert.ok(!index.includes('id="token-strip"'));
  assert.ok(!index.includes('IDEA CLAVE'));
  assert.ok(!index.includes('class="card concept-card"'));
});

test('credits identify project author and model provenance', () => {
  assert.ok(index.includes('Fernando Leon Franco'));
  assert.ok(index.includes('ChatGPT (OpenAI)'));
  assert.ok(index.includes('Familia y arquitectura GPT-2'));
  assert.ok(index.includes('mrm8488/spanish-gpt2'));
});

test('current sentence follows its latest generated content', () => {
  assert.ok(app.includes('sentence.scrollTop = sentence.scrollHeight'));
});
