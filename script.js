let start = Date.now()

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.5.1';

const input1 = document.getElementById('input1');
const input2 = document.getElementById('input2');
const generateButton = document.getElementById('generate-button');
const output = document.getElementById('output');

env.localModelPath = "/models/"
env.allowRemoteModels = false;
// env.backends.onnx.wasm.wasmPaths = '/wasm/';



const generateEmbeddings = await pipeline(
  'feature-extraction',
  'thenlper/gte-small',
  {quantized: true}
);

let timeTaken = Date.now() - start;

console.log("loaded transformers in " + timeTaken)

generateButton.disabled = false;

function dotProduct(a, b) {
  if (a.length !== b.length) {
    throw new Error('Both arguments must have the same length');
  }

  let result = 0;

  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }

  return result;
}

generateButton.addEventListener('click', async () => {
  const output1 = await generateEmbeddings(input1.value, {
    pooling: 'mean',
    normalize: true,
  });

  const output2 = await generateEmbeddings(input2.value, {
    pooling: 'mean',
    normalize: true,
  });

  const similarity = dotProduct(output1.data, output2.data);

  output.innerText = similarity;
});