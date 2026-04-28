export async function getRuntime() {
  return request('/api/runtime');
}

export async function uploadProject(file) {
  const form = new FormData();
  form.append('file', file);
  return request('/api/projects', { method: 'POST', body: form });
}

export async function analyzeProject(projectId) {
  return request(`/api/projects/${projectId}/analyze`, { method: 'POST' });
}

export async function patchComponents(projectId, payload) {
  return request(`/api/projects/${projectId}/components`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function exportPptx(projectId) {
  const response = await fetch(`/api/projects/${projectId}/export/pptx`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.blob();
}

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}
