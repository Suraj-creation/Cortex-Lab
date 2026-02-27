const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Pin the tracing root to the frontend directory so Next.js doesn't
  // get confused by the parent Cortex-Lab directory or stray lockfiles.
  outputFileTracingRoot: path.join(__dirname),

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
