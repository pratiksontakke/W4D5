// main.js - Main JavaScript file for Indian Legal Document Search System

// Theme Management
class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.createToggleButton();
    }

    createToggleButton() {
        const button = document.createElement('button');
        button.className = 'theme-toggle';
        button.innerHTML = this.theme === 'light' ? '🌙' : '☀️';
        button.addEventListener('click', () => this.toggleTheme());
        document.body.appendChild(button);
    }

    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', this.theme);
        document.documentElement.setAttribute('data-theme', this.theme);
        document.querySelector('.theme-toggle').innerHTML = this.theme === 'light' ? '🌙' : '☀️';
    }
}

// File Upload Handler
class FileUploadHandler {
    constructor(uploadAreaId, maxSizeMB = 10) {
        this.uploadArea = document.getElementById(uploadAreaId);
        this.maxSizeBytes = maxSizeMB * 1024 * 1024;
        this.allowedTypes = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain'
        ];
        this.init();
    }

    init() {
        this.setupDragAndDrop();
        this.setupFileInput();
    }

    setupDragAndDrop() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.remove('drag-over');
            });
        });

        this.uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFiles(files);
        });
    }

    setupFileInput() {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        input.accept = '.pdf,.doc,.docx,.txt';
        input.style.display = 'none';

        this.uploadArea.addEventListener('click', () => input.click());
        input.addEventListener('change', (e) => this.handleFiles(e.target.files));

        this.uploadArea.appendChild(input);
    }

    handleFiles(files) {
        Array.from(files).forEach(file => {
            if (!this.validateFile(file)) return;
            this.uploadFile(file);
        });
    }

    validateFile(file) {
        if (!this.allowedTypes.includes(file.type)) {
            this.showError('Invalid file type. Please upload PDF, DOC, DOCX, or TXT files only.');
            return false;
        }

        if (file.size > this.maxSizeBytes) {
            this.showError(`File too large. Maximum size is ${this.maxSizeBytes / 1024 / 1024}MB.`);
            return false;
        }

        return true;
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showLoading();
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const result = await response.json();
            this.showSuccess('File uploaded successfully!');
            this.triggerAnalysis(result.fileId);
        } catch (error) {
            this.showError('Failed to upload file. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-state';
        errorDiv.textContent = message;
        this.uploadArea.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    showSuccess(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-state';
        successDiv.textContent = message;
        this.uploadArea.appendChild(successDiv);
        setTimeout(() => successDiv.remove(), 5000);
    }

    showLoading() {
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        this.uploadArea.appendChild(spinner);
    }

    hideLoading() {
        const spinner = this.uploadArea.querySelector('.loading-spinner');
        if (spinner) spinner.remove();
    }

    triggerAnalysis(fileId) {
        // Trigger document analysis after successful upload
        document.dispatchEvent(new CustomEvent('documentUploaded', { detail: { fileId } }));
    }
}

// Search Handler
class SearchHandler {
    constructor(formId) {
        this.form = document.getElementById(formId);
        this.init();
    }

    init() {
        this.form.addEventListener('submit', (e) => this.handleSearch(e));
    }

    async handleSearch(e) {
        e.preventDefault();
        const query = new FormData(this.form).get('query');

        if (!query.trim()) {
            this.showError('Please enter a search query');
            return;
        }

        try {
            this.showLoading();
            const results = await this.performSearch(query);
            this.displayResults(results);
        } catch (error) {
            this.showError('Search failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async performSearch(query) {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        if (!response.ok) throw new Error('Search failed');
        return await response.json();
    }

    displayResults(results) {
        const resultsGrid = document.querySelector('.results-grid');
        resultsGrid.innerHTML = '';

        results.forEach(method => {
            const column = document.createElement('div');
            column.className = 'results-column';
            column.innerHTML = `
                <h3>${method.name}</h3>
                ${method.results.map(result => `
                    <div class="document-card">
                        <h4>${result.title}</h4>
                        <p>${result.excerpt}</p>
                        <div class="similarity-score">
                            Score: ${(result.score * 100).toFixed(1)}%
                        </div>
                    </div>
                `).join('')}
            `;
            resultsGrid.appendChild(column);
        });
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-state';
        errorDiv.textContent = message;
        this.form.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    showLoading() {
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        this.form.appendChild(spinner);
    }

    hideLoading() {
        const spinner = this.form.querySelector('.loading-spinner');
        if (spinner) spinner.remove();
    }
}

// Metrics Dashboard
class MetricsDashboard {
    constructor(dashboardId) {
        this.dashboard = document.getElementById(dashboardId);
        this.init();
    }

    init() {
        this.startPolling();
        document.addEventListener('documentUploaded', () => this.updateMetrics());
    }

    startPolling() {
        setInterval(() => this.updateMetrics(), 30000); // Update every 30 seconds
        this.updateMetrics(); // Initial update
    }

    async updateMetrics() {
        try {
            const response = await fetch('/api/metrics');
            if (!response.ok) throw new Error('Failed to fetch metrics');

            const metrics = await response.json();
            this.displayMetrics(metrics);
        } catch (error) {
            console.error('Failed to update metrics:', error);
        }
    }

    displayMetrics(metrics) {
        this.dashboard.innerHTML = `
            <div class="grid grid-cols-4">
                ${Object.entries(metrics).map(([key, value]) => `
                    <div class="metric-card">
                        <h4>${this.formatMetricName(key)}</h4>
                        <div class="metric-value">${this.formatMetricValue(value)}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    formatMetricName(key) {
        return key.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    formatMetricValue(value) {
        if (typeof value === 'number') {
            return value.toFixed(2);
        }
        return value;
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme manager
    const themeManager = new ThemeManager();

    // Initialize file upload handler
    const fileUpload = new FileUploadHandler('upload-area');

    // Initialize search handler
    const search = new SearchHandler('search-form');

    // Initialize metrics dashboard
    const metrics = new MetricsDashboard('metrics-dashboard');
});

// Export classes for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ThemeManager,
        FileUploadHandler,
        SearchHandler,
        MetricsDashboard
    };
}
