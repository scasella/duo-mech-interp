// Neurons for "Try an example": picked from a scan of 336 random neurons in layers 5-18 (next-word KL under a +-0.8
// push on "Who are you?", then greedy answers for the top ten, Hugging Face transformers in float32). These change the
// answers visibly at strength 0.8 while staying readable; the int8 engine can differ in details. `hint` only says
// where to look.
export const EXAMPLES = [
  { layer: 6, n: 1778, prompt: "Describe your perfect day.", dose: 0.8, hint: "Watch the suppressed answer." },
  { layer: 14, n: 250, prompt: "Describe your perfect day.", dose: 0.8, hint: "Compare how the boosted and suppressed answers talk." },
  { layer: 5, n: 3458, prompt: "Who are you?", dose: 0.8, hint: "Watch the boosted answer." },
];
