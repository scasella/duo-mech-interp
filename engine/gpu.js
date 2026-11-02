// Thin WebGPU layer: device setup, buffers, cached pipelines / uniforms / bind groups, one command encoder per step.
// Everything model-specific lives in model.js; every kernel is WGSL in kernels.js.

let UID = 0;

export class GPU {
  static async create() {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const L = adapter.limits;
    const want = {
      maxBufferSize: L.maxBufferSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
      maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
    };
    const device = await adapter.requestDevice({ requiredLimits: want });
    const g = new GPU(adapter, device);
    return g;
  }

  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.limits = device.limits;
    this.info = adapter.info || {};
    this.pipes = new Map();
    this.uniforms = new Map();
    this.groups = new Map();
    this.enc = null;
    this.pass = null;
    this.bytes = 0;  // live storage bytes we allocated (for the memory readout)
    this.lost = null;
    device.lost.then(info => { this.lost = info; console.error("WebGPU device lost", info); });
    device.addEventListener?.("uncapturederror", e => console.error("WebGPU:", e.error?.message || e));
  }

  buffer(size, label = "", usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    size = Math.max(16, Math.ceil(size / 16) * 16);
    const b = this.device.createBuffer({ size, usage, label });
    b.uid = ++UID;
    if (usage & GPUBufferUsage.STORAGE) this.bytes += size;
    return b;
  }

  free(b) {
    if (!b) return;
    if (b.usage & GPUBufferUsage.STORAGE) this.bytes -= b.size;
    b.destroy();
    // bind groups that reference it are dropped lazily: their keys include its uid, which is never reused
  }

  // Queue writes run before commands still sitting in an unsubmitted encoder, so flush those first: writes then land
  // in program order relative to the dispatches around them.
  write(buf, data, offset = 0) {
    this.submit();
    const view = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
    let n = view.byteLength;
    if (n % 4) {  // writeBuffer needs a multiple of 4 bytes
      const pad = new Uint8Array(Math.ceil(n / 4) * 4);
      pad.set(new Uint8Array(view.buffer, view.byteOffset, n));
      this.device.queue.writeBuffer(buf, offset, pad);
      return;
    }
    this.device.queue.writeBuffer(buf, offset, view.buffer, view.byteOffset, n);
  }

  pipeline(name, code) {
    let p = this.pipes.get(name);
    if (!p) {
      const module = this.device.createShaderModule({ code, label: name });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" }, label: name });
      p.uid = ++UID;
      this.pipes.set(name, p);
    }
    return p;
  }

  // A uniform buffer holding these words: plain numbers are u32 (negative: i32), F(x) is f32. Identical contents
  // share one buffer.
  uniform(words) {
    const key = words.map(w => (typeof w === "object" ? "f" + w.f : w)).join(",");
    let b = this.uniforms.get(key);
    if (!b) {
      const arr = new Uint32Array(Math.max(4, Math.ceil(words.length / 4) * 4));
      const f = new Float32Array(arr.buffer);
      const si = new Int32Array(arr.buffer);
      words.forEach((w, i) => { if (typeof w === "object") f[i] = w.f; else if (w < 0) si[i] = w; else arr[i] = w; });
      b = this.device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      b.uid = ++UID;
      this.device.queue.writeBuffer(b, 0, arr);
      this.uniforms.set(key, b);
      if (this.uniforms.size > 20000) { this.uniforms.clear(); this.groups.clear(); }
    }
    return b;
  }

  begin() {
    if (!this.enc) this.enc = this.device.createCommandEncoder();
    if (!this.pass) this.pass = this.enc.beginComputePass();
  }

  // bindings: [GPUBuffer | {buffer, offset, size}] in @binding order (group 0, after the params uniform at binding 0
  // when params are given).
  run(name, code, bindings, groups, params) {
    const p = this.pipeline(name, code);
    const all = params ? [this.uniform(params), ...bindings] : bindings;
    const key = p.uid + ":" + all.map(b => b.buffer ? `${b.buffer.uid}@${b.offset}+${b.size}` : b.uid).join(",");
    let bg = this.groups.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: all.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })),
      });
      this.groups.set(key, bg);
      if (this.groups.size > 50000) this.groups.clear();
    }
    this.begin();
    this.pass.setPipeline(p);
    this.pass.setBindGroup(0, bg);
    const [x, y = 1, z = 1] = Array.isArray(groups) ? groups : [groups];
    const max = this.limits.maxComputeWorkgroupsPerDimension;
    if (x > max) throw new Error(`${name}: ${x} workgroups > ${max}`);
    this.pass.dispatchWorkgroups(x, y, z);
  }

  copy(src, srcOff, dst, dstOff, size) {
    this.begin();
    this.pass.end(); this.pass = null;
    this.enc.copyBufferToBuffer(src, srcOff, dst, dstOff, size);
  }

  submit() {
    if (this.pass) { this.pass.end(); this.pass = null; }
    if (this.enc) { this.device.queue.submit([this.enc.finish()]); this.enc = null; }
  }

  async read(buf, size = buf.size, offset = 0) {
    this.submit();
    const st = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, st, 0, size);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = st.getMappedRange().slice(0);
    st.destroy();
    if (this.lost) throw new Error("WebGPU device lost: " + (this.lost.message || this.lost.reason));
    return out;
  }

  async idle() { this.submit(); await this.device.queue.onSubmittedWorkDone(); }
}

export const F = x => ({ f: x });
