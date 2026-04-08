"use strict";

/**
 * Expo config plugin scaffold for wiring native Gemma runtime artifacts.
 * This plugin is intentionally additive and no-op until native module files
 * are populated for Android/iOS targets.
 */
module.exports = function withGemmaRuntime(config) {
  return {
    ...config,
    extra: {
      ...(config.extra || {}),
      gemmaRuntime: {
        enabled: true,
      },
    },
  };
};
