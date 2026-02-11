<template>
  <div class="p-8 max-w-4xl mx-auto font-sans">
    <h1 class="text-3xl font-bold mb-6 text-blue-600">Audio Embedding Studio</h1>
    
    <nav class="flex gap-4 mb-8 border-b">
      <button @click="view = 'upload'" :class="navClass('upload')">Upload</button>
      <button @click="handleNav('dashboard')" :class="navClass('dashboard')">Dashboard</button>
      <button @click="view = 'search'" :class="navClass('search')">Search</button>
    </nav>

    <div v-if="error" class="mb-4 p-4 bg-red-100 text-red-700 rounded flex justify-between">
      <span>{{ error }}</span>
      <button @click="error = null" class="font-bold">&times;</button>
    </div>

    <div v-if="view === 'upload'" class="bg-gray-50 p-6 rounded-lg shadow-sm">
      <h2 class="text-xl font-semibold mb-4">Upload Audio (Max 10)</h2>
      <input type="file" multiple @change="handleFiles" accept=".wav,.mp3" class="mb-4 block w-full text-sm" />
      <button @click="uploadFiles" :disabled="loading || files.length === 0" class="w-full bg-blue-600 text-white py-2 rounded disabled:opacity-50">
        {{ loading ? 'Processing...' : 'Upload' }}
      </button>
    </div>

    <div v-if="view === 'dashboard'">
      <button @click="clearDatabase" class="mb-4 text-xs bg-red-50 text-red-600 px-2 py-1 rounded border border-red-200">Clear All</button>
      <div v-if="loading" class="text-center py-10">Loading...</div>
      <div v-for="file in allFiles" :key="file.id" class="p-4 bg-white border rounded-lg mb-2 flex justify-between items-center shadow-sm">
        <div>
          <p class="font-medium">{{ file.filename }}</p>
          <p class="text-xs text-gray-400 font-mono">{{ file.id.slice(0,8) }}</p>
        </div>
        <div class="flex gap-4">
          <button @click="prepSearch(file)" class="text-blue-600 text-sm">Search Similar</button>
          <button @click="deleteFile(file.id)" class="text-red-500 text-sm">Delete</button>
        </div>
      </div>
    </div>

    <div v-if="view === 'search'">
      <div class="bg-blue-50 p-6 rounded-lg mb-6 border border-blue-100">
        <h2 class="text-xl font-semibold mb-4 text-blue-800">Vector Search</h2>
        <div class="flex gap-4 mb-4">
          <input type="file" @change="e => searchFile = e.target.files[0]" accept=".wav,.mp3" class="block w-full text-sm" />
          <div class="w-24">
            <label class="block text-[10px] uppercase font-bold text-gray-400">K-Value</label>
            <input type="number" v-model="topK" min="1" max="20" class="w-full p-1 border rounded" />
          </div>
        </div>
        <button @click="performSearch" :disabled="!searchFile || loading" class="bg-green-600 text-white font-bold py-2 px-6 rounded disabled:opacity-50">Search</button>
      </div>
      
      <div v-if="currentSearchName && results.length" class="mb-4 text-sm text-gray-500 italic">
        Matches for: <strong>{{ currentSearchName }}</strong>
      </div>

      <div v-for="(res, i) in results" :key="res.id" class="p-4 mb-2 bg-white border rounded flex justify-between items-center shadow-sm">
        <span><span class="text-gray-300 mr-2">#{{i+1}}</span>{{ res.filename }}</span>
        <span class="font-mono font-bold text-blue-600">{{ (res.similarity_score * 100).toFixed(1) }}%</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const view = ref('upload');
const loading = ref(false);
const error = ref(null);
const files = ref([]);
const allFiles = ref([]);
const searchFile = ref(null);
const results = ref([]);
const topK = ref(5);
const currentSearchName = ref("");

const navClass = (v) => `pb-2 px-4 ${view.value === v ? 'border-b-2 border-blue-500 text-blue-600 font-bold' : 'text-gray-500'}`;

const handleNav = (v) => { view.value = v; if (v === 'dashboard') fetchFiles(); };

const handleFiles = (e) => {
  const selected = Array.from(e.target.files);
  files.value = selected.slice(0, 10);
  if (selected.length > 10) error.value = "Limited to 10 files.";
};

const uploadFiles = async () => {
  loading.value = true; error.value = null;
  const formData = new FormData();
  files.value.forEach(f => formData.append('files', f));
  try {
    const res = await axios.post(`${API}/embeddings`, formData);
    alert("Upload complete.");
    files.value = [];
  } catch (err) { error.value = "Upload failed."; }
  finally { loading.value = false; }
};

const fetchFiles = async () => {
  loading.value = true;
  try { const res = await axios.get(`${API}/embeddings`); allFiles.value = res.data; }
  catch (err) { error.value = "Failed to fetch data."; }
  finally { loading.value = false; }
};

const prepSearch = async (file) => {
  view.value = 'search'; loading.value = true; results.value = []; currentSearchName.value = file.filename;
  try {
    const res = await axios.post(`${API}/search-by-id/${file.id}?top_k=${topK.value}`);
    results.value = res.data;
  } catch (err) { error.value = "Search failed."; }
  finally { loading.value = false; }
};

const performSearch = async () => {
  loading.value = true; error.value = null; currentSearchName.value = searchFile.value.name;
  const formData = new FormData();
  formData.append('file', searchFile.value);
  formData.append('top_k', topK.value);
  try {
    const res = await axios.post(`${API}/search`, formData);
    results.value = res.data;
  } catch (err) { error.value = "Search failed."; }
  finally { loading.value = false; }
};

const deleteFile = async (id) => {
  if (!confirm("Delete?")) return;
  await axios.delete(`${API}/embeddings/${id}`);
  fetchFiles();
};

const clearDatabase = async () => {
  if (!confirm("Wipe all?")) return;
  await axios.delete(`${API}/embeddings`);
  allFiles.value = [];
};
</script>