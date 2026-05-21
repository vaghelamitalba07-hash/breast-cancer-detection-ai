const API_BASE = "/api";

function getToken() {
  return localStorage.getItem("token");
}

function getUsername() {
  return localStorage.getItem("username");
}

function setAuth(token, username) {
  localStorage.setItem("token", token);
  localStorage.setItem("username", username);
}

function clearAuth() {
  localStorage.removeItem("token");
  localStorage.removeItem("username");
}

function requireAuth() {
  if (!getToken()) {
    window.location.href = "/login";
    return false;
  }
  return true;
}

async function apiFetch(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));

  if (res.status === 401 && !path.startsWith("/auth/")) {
    clearAuth();
    window.location.href = "/login";
    throw new Error("Session expired");
  }
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Request failed");
  }
  return data;
}
