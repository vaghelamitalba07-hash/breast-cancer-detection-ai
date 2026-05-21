async function handleLogin(e) {
  e.preventDefault();
  const errEl = document.getElementById("error");
  errEl.textContent = "";
  errEl.classList.add("hidden");

  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;

  try {
    const data = await apiFetch("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    setAuth(data.token, data.username);
    window.location.href = "/dashboard";
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
  }
}

async function handleSignup(e) {
  e.preventDefault();
  const errEl = document.getElementById("error");
  errEl.textContent = "";
  errEl.classList.add("hidden");

  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const confirm = document.getElementById("confirm").value;

  if (password !== confirm) {
    errEl.textContent = "Passwords do not match.";
    errEl.classList.remove("hidden");
    return;
  }

  try {
    const data = await apiFetch("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    setAuth(data.token, data.username);
    window.location.href = "/dashboard";
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
  }
}

async function handleLogout() {
  try {
    await apiFetch("/auth/logout", { method: "POST" });
  } catch (_) {}
  clearAuth();
  window.location.href = "/";
}

function demoLogin() {
  document.getElementById("username").value = "demo";
  document.getElementById("password").value = "demo123";
  document.getElementById("login-form").dispatchEvent(new Event("submit"));
}
