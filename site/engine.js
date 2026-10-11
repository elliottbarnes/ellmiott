// A deliberately readable inference engine for the exact PyTorch model in this repo.
// Matrix weights are row-major [output, input], as in torch.nn.Linear.
export const FORMAT = 'ellmiott-gpt-v1';
const MAX_PARAMETERS = 2_000_000;

export function softmax(values) {
  const max = Math.max(...values);
  const exps = Array.from(values, value => Math.exp(value - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(value => value / sum);
}

export function distribution(logits, temperature = 1, topK = 0) {
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 3)
    throw new Error('Temperature must be between 0 and 3.');
  if (!Number.isInteger(topK) || topK < 0 || topK > logits.length)
    throw new Error('Top-k must be zero (all tokens) or a vocabulary-sized positive integer.');
  if (temperature === 0) {
    const best = logits.indexOf(Math.max(...logits));
    return logits.map((_, index) => Number(index === best));
  }
  const ranked = logits.map((value, id) => ({value, id})).sort((a, b) => b.value - a.value || a.id - b.id);
  const allowed = new Set(ranked.slice(0, topK || logits.length).map(item => item.id));
  // Centre before scaling so very small temperatures cannot overflow the winner.
  const max = ranked[0].value;
  return softmax(logits.map((value, id) => allowed.has(id) ? (value-max) / temperature : -Infinity));
}

export function randomSource(seed = 42) {
  let state = Number(seed) >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function sample(probabilities, random = Math.random) {
  const threshold = random();
  let sum = 0;
  let lastNonzero = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > 0) lastNonzero = i;
    sum += probabilities[i];
    if (threshold < sum) return i;
  }
  return lastNonzero;
}

function expectedShapes(config) {
  const {vocab_size: V, n_embd: C, block_size: T, n_layer: L} = config;
  const shapes = {'token_embedding.weight': [V, C], 'position_embedding.weight': [T, C],
    'ln_f.weight': [C], 'ln_f.bias': [C], 'lm_head.weight': [V, C]};
  for (let i = 0; i < L; i++) {
    const p = `blocks.${i}`;
    for (const norm of ['ln1', 'ln2']) {
      shapes[`${p}.${norm}.weight`] = [C]; shapes[`${p}.${norm}.bias`] = [C];
    }
    for (const [name, output, input] of [['attn.qkv', 3*C, C], ['attn.proj', C, C], ['mlp.fc', 4*C, C], ['mlp.proj', C, 4*C]]) {
      shapes[`${p}.${name}.weight`] = [output, input]; shapes[`${p}.${name}.bias`] = [output];
    }
  }
  return shapes;
}

export class TinyGPT {
  constructor(checkpoint) {
    if (checkpoint?.format !== FORMAT) throw new Error('Unsupported checkpoint format.');
    const c = checkpoint.config;
    if (!c || !['vocab_size', 'block_size', 'n_embd', 'n_head', 'n_layer'].every(k => Number.isInteger(c[k]) && c[k] > 0)
      || c.vocab_size > 256 || c.block_size > 128 || c.n_embd > 128 || c.n_layer > 4 || c.n_embd % c.n_head)
      throw new Error('Unsupported model dimensions.');
    if (!Array.isArray(checkpoint.vocab) || checkpoint.vocab.length !== c.vocab_size
      || new Set(checkpoint.vocab).size !== c.vocab_size
      || checkpoint.vocab.some(char => typeof char !== 'string' || Array.from(char).length !== 1))
      throw new Error('Invalid character vocabulary.');
    this.config = {...c}; this.vocab = [...checkpoint.vocab];
    this.charIds = new Map(this.vocab.map((char, id) => [char, id]));
    this.weights = {}; let total = 0;
    const shapes = expectedShapes(c);
    if (Object.keys(checkpoint.tensors ?? {}).length !== Object.keys(shapes).length)
      throw new Error('Checkpoint has missing or unexpected tensors.');
    for (const [name, shape] of Object.entries(shapes)) {
      const tensor = checkpoint.tensors?.[name];
      const count = shape.reduce((a, b) => a*b, 1); total += count;
      if (total > MAX_PARAMETERS || !tensor || JSON.stringify(tensor.shape) !== JSON.stringify(shape)
        || !Array.isArray(tensor.data) || tensor.data.length !== count
        || tensor.data.some(value => !Number.isFinite(value) || Math.abs(value) > 1e4))
        throw new Error(`Invalid checkpoint tensor: ${name}`);
      this.weights[name] = Float32Array.from(tensor.data);
    }
    this.parameterCount = total;
  }

  tokenize(text) {
    const chars = Array.from(String(text || (this.charIds.has('\n') ? '\n' : this.vocab[0])));
    const fallback = this.charIds.get('?') ?? this.charIds.get(' ') ?? 0;
    const truncated = Math.max(0, chars.length - this.config.block_size);
    let unknown = 0;
    const ids = chars.slice(-this.config.block_size).map(char => {
      const id = this.charIds.get(char);
      if (id === undefined) { unknown++; return fallback; }
      return id;
    });
    return {ids, labels: ids.map(id => this.vocab[id]), truncated, unknown};
  }

  linear(input, rows, inputWidth, name, outputWidth, bias = true) {
    const w = this.weights[`${name}.weight`]; const b = this.weights[`${name}.bias`];
    const result = new Float32Array(rows * outputWidth);
    for (let r = 0; r < rows; r++) for (let o = 0; o < outputWidth; o++) {
      let sum = bias ? b[o] : 0;
      const xBase = r * inputWidth; const wBase = o * inputWidth;
      for (let k = 0; k < inputWidth; k++) sum += input[xBase+k] * w[wBase+k];
      result[r * outputWidth + o] = sum;
    }
    return result;
  }

  norm(input, rows, name) {
    const C = this.config.n_embd;
    const w = this.weights[`${name}.weight`], b = this.weights[`${name}.bias`];
    const result = new Float32Array(input.length);
    for (let r = 0; r < rows; r++) {
      const offset = r*C; let mean = 0; let variance = 0;
      for (let j = 0; j < C; j++) mean += input[offset+j];
      mean /= C;
      for (let j = 0; j < C; j++) variance += (input[offset+j] - mean) ** 2;
      const scale = 1 / Math.sqrt(variance/C + 1e-5);
      for (let j = 0; j < C; j++) result[offset+j] = (input[offset+j] - mean)*scale*w[j] + b[j];
    }
    return result;
  }

  forward(ids, {layer = 0, head = 0, disabledHead = null} = {}) {
    const {n_embd: C, n_head: H, n_layer: L, block_size: maxT} = this.config;
    if (!Array.isArray(ids) || !ids.length || ids.length > maxT || ids.some(id => !Number.isInteger(id) || id < 0 || id >= this.vocab.length))
      throw new Error('Invalid input token sequence.');
    if (!Number.isInteger(layer) || layer < 0 || layer >= L || !Number.isInteger(head) || head < 0 || head >= H)
      throw new Error('Invalid attention layer or head.');
    if (disabledHead && (!Number.isInteger(disabledHead.layer) || disabledHead.layer < 0 || disabledHead.layer >= L
      || !Number.isInteger(disabledHead.head) || disabledHead.head < 0 || disabledHead.head >= H))
      throw new Error('Invalid head ablation.');
    const T = ids.length, D = C/H;
    const token = this.weights['token_embedding.weight'], position = this.weights['position_embedding.weight'];
    const x = new Float32Array(T*C);
    const embeddings = [];
    for (let r = 0; r < T; r++) {
      embeddings.push(Array.from(token.slice(ids[r]*C, ids[r]*C + 8)));
      for (let j = 0; j < C; j++) x[r*C+j] = token[ids[r]*C+j] + position[r*C+j];
    }
    let selectedAttention = [];
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}`;
      const qkv = this.linear(this.norm(x, T, `${p}.ln1`), T, C, `${p}.attn.qkv`, 3*C);
      const context = new Float32Array(T*C);
      for (let h = 0; h < H; h++) for (let r = 0; r < T; r++) {
        const scores = [];
        for (let s = 0; s <= r; s++) {
          let score = 0;
          for (let j = 0; j < D; j++) score += qkv[r*3*C + h*D+j] * qkv[s*3*C + C + h*D+j];
          scores.push(score / Math.sqrt(D));
        }
        const probs = softmax(scores);
        if (l === layer && h === head) selectedAttention.push([...probs, ...Array(T-r-1).fill(0)]);
        if (disabledHead?.layer === l && disabledHead?.head === h) continue;
        for (let j = 0; j < D; j++) {
          let sum = 0;
          for (let s = 0; s <= r; s++) sum += probs[s] * qkv[s*3*C + 2*C + h*D+j];
          context[r*C+h*D+j] = sum;
        }
      }
      const attn = this.linear(context, T, C, `${p}.attn.proj`, C);
      for (let j = 0; j < x.length; j++) x[j] += attn[j];
      const hidden = this.linear(this.norm(x, T, `${p}.ln2`), T, C, `${p}.mlp.fc`, 4*C);
      for (let j = 0; j < hidden.length; j++) {
        const v = hidden[j]; hidden[j] = 0.5*v*(1 + Math.tanh(Math.sqrt(2/Math.PI)*(v + 0.044715*v*v*v)));
      }
      const mlp = this.linear(hidden, T, 4*C, `${p}.mlp.proj`, C);
      for (let j = 0; j < x.length; j++) x[j] += mlp[j];
    }
    const normalized = this.norm(x, T, 'ln_f');
    const logits = Array.from(this.linear(normalized.slice((T-1)*C), 1, C, 'lm_head', this.vocab.length, false));
    return {logits, attention: selectedAttention, embeddings};
  }

  inspect(text, options = {}) {
    const tokens = this.tokenize(text);
    const result = this.forward(tokens.ids, options);
    return {...result, tokens, baseProbabilities: softmax(result.logits),
      probabilities: distribution(result.logits, options.temperature ?? 1, options.topK ?? 0),
      layer: options.layer ?? 0, head: options.head ?? 0};
  }
}
