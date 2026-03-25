(function () {
  // Replace with your actual Render backend URL after first deployment.
  var RENDER_API_BASE = "https://mini-project--sridevisri2.replit.app";

  var host = window.location.hostname;
  var isLocal = host === "127.0.0.1" || host === "localhost";
  var isGithubPages = host.endsWith("github.io");

  var apiBase = "";
  if (isLocal) {
    apiBase = "http://127.0.0.1:5000";
  } else if (isGithubPages) {
    apiBase = RENDER_API_BASE;
  }

  if (apiBase.endsWith("/")) {
    apiBase = apiBase.slice(0, -1);
  }

  window.CPA_API_BASE = apiBase;
  window.CPA_API_NEEDS_RENDER_URL = isGithubPages && RENDER_API_BASE.indexOf("YOUR-RENDER-SERVICE") >= 0;
})();
