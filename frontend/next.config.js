const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Pin the tracing root to the frontend directory so Next.js doesn't
  // get confused by the parent Cortex-Lab directory or stray lockfiles.
  outputFileTracingRoot: path.join(__dirname),

  // Increase proxy timeout for long-running RAG streaming requests
  // The RAG pipeline can take 30-60s for retrieval before streaming starts
  experimental: {
    proxyTimeout: 300000,  // 5 minutes
  },

  // Keep HTTP connections alive for SSE streaming
  httpAgentOptions: {
    keepAlive: true,
  },

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },

  // Allow the backend origin for images / assets if ever needed
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Connection", value: "keep-alive" },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
