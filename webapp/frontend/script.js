class FaceRatingApp {
    constructor() {
        this.API_BASE = 'http://localhost:8000'; // Change to your deployed API URL
        this.sessionId = this.generateSessionId();
        this.currentImageId = null;
        this.imagesRated = 0;
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadRandomImage();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        this.elements = {
            loading: document.getElementById('loading'),
            ratingSection: document.getElementById('rating-section'),
            faceImage: document.getElementById('face-image'),
            ratingSlider: document.getElementById('rating-slider'),
            ratingValue: document.getElementById('rating-value'),
            submitBtn: document.getElementById('submit-rating'),
            successMessage: document.getElementById('success-message'),
            nextBtn: document.getElementById('next-image'),
            imagesRatedSpan: document.getElementById('images-rated'),
            viewStatsBtn: document.getElementById('view-stats'),
            statsModal: document.getElementById('stats-modal'),
            statsContent: document.getElementById('stats-content'),
            closeModal: document.querySelector('.close')
        };
    }
    
    setupEventListeners() {
        // Rating slider
        this.elements.ratingSlider.addEventListener('input', (e) => {
            this.elements.ratingValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        
        // Submit rating
        this.elements.submitBtn.addEventListener('click', () => {
            this.submitRating();
        });
        
        // Next image
        this.elements.nextBtn.addEventListener('click', () => {
            this.loadRandomImage();
        });
        
        // View stats
        this.elements.viewStatsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.showStats();
        });
        
        // Close modal
        this.elements.closeModal.addEventListener('click', () => {
            this.elements.statsModal.style.display = 'none';
        });
        
        // Close modal on outside click
        window.addEventListener('click', (e) => {
            if (e.target === this.elements.statsModal) {
                this.elements.statsModal.style.display = 'none';
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && this.elements.ratingSection.style.display !== 'none') {
                this.submitRating();
            }
            if (e.key === 'n' && this.elements.successMessage.style.display !== 'none') {
                this.loadRandomImage();
            }
        });
    }
    
    showLoading() {
        this.elements.loading.style.display = 'block';
        this.elements.ratingSection.style.display = 'none';
        this.elements.successMessage.style.display = 'none';
    }
    
    showRatingSection() {
        this.elements.loading.style.display = 'none';
        this.elements.ratingSection.style.display = 'block';
        this.elements.successMessage.style.display = 'none';
        this.elements.submitBtn.disabled = false;
    }
    
    showSuccessMessage() {
        this.elements.loading.style.display = 'none';
        this.elements.ratingSection.style.display = 'none';
        this.elements.successMessage.style.display = 'block';
    }
    
    async loadRandomImage() {
        this.showLoading();
        
        try {
            const response = await fetch(`${this.API_BASE}/random-image?session_id=${this.sessionId}`);
            const data = await response.json();
            
            if (data.success) {
                this.currentImageId = data.image_id;
                this.elements.faceImage.src = `${this.API_BASE}${data.image_url}`;
                this.elements.faceImage.onload = () => {
                    this.showRatingSection();
                };
                
                // Reset rating slider
                this.elements.ratingSlider.value = 5;
                this.elements.ratingValue.textContent = '5.0';
            } else {
                throw new Error(data.detail || 'Failed to load image');
            }
        } catch (error) {
            console.error('Error loading image:', error);
            this.elements.loading.innerHTML = `
                <p>❌ Failed to load image</p>
                <button onclick="location.reload()">Retry</button>
            `;
        }
    }
    
    async submitRating() {
        if (!this.currentImageId) return;
        
        this.elements.submitBtn.disabled = true;
        this.elements.submitBtn.textContent = 'Submitting...';
        
        const rating = parseFloat(this.elements.ratingSlider.value);
        
        try {
            const response = await fetch(`${this.API_BASE}/rate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_id: this.currentImageId,
                    human_rating: rating,
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.imagesRated++;
                this.elements.imagesRatedSpan.textContent = this.imagesRated;
                this.showSuccessMessage();
            } else {
                throw new Error(data.detail || 'Failed to submit rating');
            }
        } catch (error) {
            console.error('Error submitting rating:', error);
            alert('Failed to submit rating. Please try again.');
            this.elements.submitBtn.disabled = false;
            this.elements.submitBtn.textContent = 'Submit Rating';
        }
    }
    
    async showStats() {
        this.elements.statsModal.style.display = 'block';
        this.elements.statsContent.innerHTML = 'Loading...';
        
        try {
            const response = await fetch(`${this.API_BASE}/stats`);
            const data = await response.json();
            
            this.elements.statsContent.innerHTML = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-value">${data.total_ratings}</span>
                        <div class="stat-label">Total Ratings</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">${data.unique_images}</span>
                        <div class="stat-label">Unique Images</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">${data.average_rating}</span>
                        <div class="stat-label">Average Rating</div>
                    </div>
                </div>
                
                <h3>Rating Distribution</h3>
                <div class="rating-distribution">
                    ${Object.entries(data.rating_distribution || {})
                        .map(([rating, count]) => 
                            `<div>${rating} stars: ${count} ratings</div>`
                        ).join('')}
                </div>
                
                ${data.recent_ratings && data.recent_ratings.length > 0 ? `
                    <h3>Recent Ratings</h3>
                    <div class="recent-ratings">
                        ${data.recent_ratings.slice(-5).map(rating => 
                            `<div>${rating.image_id}: ${rating.human_rating}/10</div>`
                        ).join('')}
                    </div>
                ` : ''}
            `;
        } catch (error) {
            console.error('Error loading stats:', error);
            this.elements.statsContent.innerHTML = 'Failed to load statistics.';
        }
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new FaceRatingApp();
});