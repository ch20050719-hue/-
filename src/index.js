// 口罩检测系统 - Cloudflare Worker
// 托管 public/ 静态资源
export default {
  async fetch(request, env) {
    // 直接由 Cloudflare Assets 托管 public/ 目录
    return env.ASSETS.fetch(request);
  },
};
