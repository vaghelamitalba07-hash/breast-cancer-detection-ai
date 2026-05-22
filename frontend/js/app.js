/* ========================================
   ONCOSENSE AI - GLOBAL FUNCTIONS
   ======================================== */

// ========== AUTHENTICATION ==========
let authToken = localStorage.getItem('token');
let currentUser = localStorage.getItem('username');

function checkAuth() {
    if (!authToken) {
        window.location.href = '/login';
        return false;
    }
    return true;
}

async function logout() {
    try {
        await fetch('/api/auth/logout', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
    } catch(e) {}
    localStorage.clear();
    window.location.href = '/';
}

// ========== API CALLS ==========
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`
        }
    };
    if (body) options.body = JSON.stringify(body);
    
    const res = await fetch(endpoint, options);
    if (res.status === 401) {
        alert('Session expired. Please login again.');
        localStorage.clear();
        window.location.href = '/login';
        return null;
    }
    return res.json();
}

// ========== SET ACTIVE LINK ==========
function setActiveLink() {
    const currentPath = window.location.pathname;
    document.querySelectorAll('.sidebar a').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// ========== DONATION ==========
function openDonateModal(amount) {
    const modal = document.getElementById('donationModal');
    if (amount) document.getElementById('donorAmount').value = amount;
    if (modal) modal.style.display = 'flex';
}

function closeDonateModal() {
    const modal = document.getElementById('donationModal');
    if (modal) modal.style.display = 'none';
}

async function processDonation() {
    const name = document.getElementById('donorName')?.value || 'Anonymous';
    const email = document.getElementById('donorEmail')?.value;
    const amount = parseFloat(document.getElementById('donorAmount')?.value);
    const message = document.getElementById('donorMessage')?.value || '';
    
    if (!email || !amount) {
        alert('Please fill email and amount');
        return;
    }
    
    alert(`Thank you ${name} for ₹${amount}! Certificate will be downloaded.`);
    closeDonateModal();
    
    try {
        const res = await fetch('/generate-certificate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, amount, message, email })
        });
        if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `OncoSense_Certificate_${name.replace(/\s+/g, '_')}.pdf`;
            a.click();
            URL.revokeObjectURL(url);
        }
    } catch(e) {}
    
    // Clear form
    ['donorName', 'donorEmail', 'donorAmount', 'donorMessage'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
}

// ========== LOADER ==========
function showLoader(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = `<div class="loader"><div class="spinner"></div><p>Loading...</p></div>`;
    }
}

function hideLoader(elementId, content) {
    const el = document.getElementById(elementId);
    if (el) el.innerHTML = content;
}