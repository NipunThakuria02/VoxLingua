class VoxLinguaCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(1024);
    this.used = 0;
  }

  process(inputs) {
    const samples = inputs[0]?.[0];
    if (!samples) return true;
    for (const sample of samples) {
      this.block[this.used++] = sample;
      if (this.used === this.block.length) {
        this.port.postMessage(this.block, [this.block.buffer]);
        this.block = new Float32Array(1024);
        this.used = 0;
      }
    }
    // Leave the output silent while the connected graph pulls the microphone input.
    return true;
  }
}

registerProcessor('voxlingua-capture', VoxLinguaCapture);
