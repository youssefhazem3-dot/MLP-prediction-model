import { MODEL_WEIGHTS } from "./model-weights.js";

const examples = {
  high: { recency: 8, monetary: 1250, avgQuantity: 12 },
  steady: { recency: 42, monetary: 480, avgQuantity: 5 },
  cold: { recency: 120, monetary: 0, avgQuantity: 1 },
};

const form = document.querySelector("#prediction-form");
const fields = {
  recency: document.querySelector("#recency"),
  monetary: document.querySelector("#monetary"),
  avgQuantity: document.querySelector("#avgQuantity"),
};
const probabilityRing = document.querySelector("#probability-ring");
const probabilityValue = document.querySelector("#probability-value");
const predictionLabel = document.querySelector("#prediction-label");
const predictionDetail = document.querySelector("#prediction-detail");
const confidenceValue = document.querySelector("#confidence-value");
const confidenceBar = document.querySelector("#confidence-bar");

function relu(value) {
  return Math.max(0, value);
}

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }

  const z = Math.exp(value);
  return z / (1 + z);
}

function activate(value, activation) {
  if (activation === "relu") return relu(value);
  if (activation === "sigmoid") return sigmoid(value);
  return value;
}

function dense(input, layer) {
  return layer.bias.map((bias, columnIndex) => {
    const weightedSum = input.reduce((sum, value, rowIndex) => {
      return sum + value * layer.kernel[rowIndex][columnIndex];
    }, bias);

    return activate(weightedSum, layer.activation);
  });
}

function predict(features) {
  const output = MODEL_WEIGHTS.layers.reduce((layerInput, layer) => dense(layerInput, layer), features);
  return output[0];
}

function readFeatures() {
  return MODEL_WEIGHTS.inputFeatures.map((feature) => {
    const value = Number(fields[feature].value);
    return Number.isFinite(value) && value > 0 ? value : 0;
  });
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderPrediction() {
  const probability = Math.min(1, Math.max(0, predict(readFeatures())));
  const confidence = Math.abs(probability - MODEL_WEIGHTS.threshold) * 2;
  const isFrequent = probability >= MODEL_WEIGHTS.threshold;
  const ringValue = Math.round(probability * 100);

  probabilityRing.style.setProperty("--probability", `${ringValue}%`);
  probabilityValue.textContent = `${ringValue}%`;
  predictionLabel.textContent = isFrequent ? "Frequent customer likely" : "Frequent customer unlikely";
  predictionDetail.textContent = isFrequent
    ? "The customer profile is above the model threshold for repeat purchase frequency."
    : "The customer profile is below the model threshold for repeat purchase frequency.";
  confidenceValue.textContent = formatPercent(confidence);
  confidenceBar.style.width = formatPercent(confidence);
  probabilityRing.dataset.state = isFrequent ? "positive" : "negative";
}

function applyExample(name) {
  const example = examples[name];
  if (!example) return;

  Object.entries(example).forEach(([key, value]) => {
    fields[key].value = value;
  });

  renderPrediction();
}

Object.values(fields).forEach((field) => {
  field.addEventListener("input", renderPrediction);
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => applyExample(button.dataset.example));
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  renderPrediction();
});

form.addEventListener("reset", () => {
  window.setTimeout(renderPrediction, 0);
});

renderPrediction();
