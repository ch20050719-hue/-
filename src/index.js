// 口罩检测系统 - Cloudflare Worker
// 托管 public/ 静态资源 + 代理 API 到后端隧道
const BACKEND_TUNNEL = 'https://b2d52e0bb5fbeb.lhr.life';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // API 请求代理到后端隧道
    if (path === '/predict' || path === '/predict_video' || path === '/health') {
      const backendUrl = BACKEND_TUNNEL + path;
      const proxyRequest = new Request(backendUrl, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return fetch(proxyRequest);
    }

    // 静态资源由 assets 托管
    return env.ASSETS.fetch(request);
  },
};
