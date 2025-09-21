/**
 * GameMatch Dashboard - Industry Standard JavaScript
 * Modern ES6+ code with proper error handling and performance optimization
 */

class GameMatchDashboard {
    constructor() {
        this.apiBase = '/api';
        this.currentQuery = '';
        this.isLoading = false;
        this.charts = {};
        
        this.init();
    }
    
    init() {
        console.log('Initializing GameMatch Dashboard...');
        this.debugElements();
        this.bindEvents();
        this.loadInitialData();
    }
    
    debugElements() {
        console.log('Checking for required elements:');
        console.log('- Search form:', document.getElementById('searchForm'));
        console.log('- Search input:', document.getElementById('searchInput'));
        console.log('- Stats container:', document.getElementById('statsContainer'));
        console.log('- Games section:', document.getElementById('gamesSection'));
        console.log('- Games grid:', document.querySelector('.games-grid'));
        console.log('- Genre chart:', document.getElementById('genreChart'));
        console.log('- Price chart:', document.getElementById('priceChart'));
        console.log('- Rating chart:', document.getElementById('ratingChart'));
    }
    
    bindEvents() {
        // Search form submission
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleSearch();
            });
        }
        
        // Search input enter key
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSearch();
                }
            });
        }
        
        // Filter change events
        const genreFilter = document.getElementById('genreFilter');
        const priceFilter = document.getElementById('priceFilter');
        const ratingFilter = document.getElementById('ratingFilter');
        
        if (genreFilter) {
            genreFilter.addEventListener('change', () => this.handleFilters());
        }
        if (priceFilter) {
            priceFilter.addEventListener('change', () => this.handleFilters());
        }
        if (ratingFilter) {
            ratingFilter.addEventListener('change', () => this.handleFilters());
        }
        
        // Clear filters button
        const clearFilters = document.getElementById('clearFilters');
        if (clearFilters) {
            clearFilters.addEventListener('click', () => this.clearFilters());
        }
    }
    
    handleFilters() {
        const searchInput = document.getElementById('searchInput');
        const query = searchInput.value.trim();
        
        // If there's a search query, apply filters to search results
        if (query) {
            this.handleSearch();
        } else {
            // If no search query, show filtered popular games
            this.loadFilteredGames();
        }
    }
    
    loadFilteredGames() {
        const genreFilter = document.getElementById('genreFilter').value;
        const priceFilter = document.getElementById('priceFilter').value;
        const ratingFilter = document.getElementById('ratingFilter').value;
        
        let query = 'popular';
        
        // Build query based on filters
        if (genreFilter) {
            query = genreFilter;
        }
        
        this.searchGames(query, 50);
    }
    
    clearFilters() {
        document.getElementById('genreFilter').value = '';
        document.getElementById('priceFilter').value = '';
        document.getElementById('ratingFilter').value = '';
        this.loadSampleGames();
    }
    
    loadInitialData() {
        Promise.all([
            this.loadStats(),
            this.loadSampleGames()
        ]).catch(error => {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load dashboard data');
        });
    }
    
    loadStats() {
        console.log('Loading stats from:', this.apiBase + '/stats');
        
        fetch(this.apiBase + '/stats')
            .then(response => {
                console.log('Stats response status:', response.status);
                if (!response.ok) {
                    throw new Error('HTTP error! status: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                console.log('Stats data received:', data);
                console.log('Total games:', data.total_games);
                
                this.renderStats(data);
                this.statsData = data;
                this.attemptChartCreation();
            })
            .catch(error => {
                console.error('Error loading stats:', error);
                const statsContainer = document.getElementById('statsContainer');
                if (statsContainer) {
                    statsContainer.innerHTML = '<div class="error-message" style="color: red; padding: 20px; text-align: center;"><i class="fas fa-exclamation-triangle"></i> Failed to load statistics: ' + error.message + '<br><button onclick="dashboardInstance.loadStats()" style="margin-top: 10px; padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Retry</button></div>';
                }
            });
    }
    
    attemptChartCreation() {
        console.log('Attempting chart creation. Chart available:', typeof Chart !== 'undefined', 'ChartJS Ready:', window.chartJsReady, 'Stats data:', !!this.statsData);
        
        // Check both Chart availability and our global flag
        if ((typeof Chart !== 'undefined' || window.chartJsReady) && this.statsData) {
            console.log('Chart.js is available, creating charts...');
            this.createCharts(this.statsData);
        } else {
            console.log('Chart.js not available or no stats data, retrying...');
            
            // Set up a more aggressive retry mechanism
            let attempts = 0;
            const maxAttempts = 10;
            
            const retryCharts = () => {
                attempts++;
                console.log('Chart creation attempt', attempts, 'of', maxAttempts);
                console.log('- Chart available:', typeof Chart !== 'undefined');
                console.log('- ChartJS Ready flag:', window.chartJsReady);
                console.log('- Stats data available:', !!this.statsData);
                
                if ((typeof Chart !== 'undefined' || window.chartJsReady) && this.statsData) {
                    console.log('Success! Creating charts on attempt', attempts);
                    this.createCharts(this.statsData);
                    return;
                }
                
                if (attempts < maxAttempts) {
                    setTimeout(retryCharts, 500); // Faster retries
                } else {
                    console.error('Failed to create charts after', maxAttempts, 'attempts');
                    // Try one more time with direct Chart access
                    if (this.statsData) {
                        console.log('Final attempt: trying to create charts anyway...');
                        try {
                            this.createCharts(this.statsData);
                        } catch (error) {
                            console.error('Final chart creation failed:', error);
                            this.showChartError();
                        }
                    } else {
                        this.showChartError();
                    }
                }
            };
            
            setTimeout(retryCharts, 500);
        }
    }
    
    showChartError() {
        console.log('Showing chart error message');
        const chartsSection = document.querySelector('.games-section h3');
        if (chartsSection && chartsSection.textContent.includes('Analytics Dashboard')) {
            chartsSection.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Analytics Dashboard (Charts failed to load - <button onclick="refreshCharts()" style="background:none;border:none;color:var(--primary-color);text-decoration:underline;cursor:pointer;">Retry</button>)';
        } else {
            // Try to find any element with "Analytics" in it
            const allElements = document.querySelectorAll('*');
            for (let element of allElements) {
                if (element.textContent && element.textContent.includes('Analytics Dashboard')) {
                    element.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Analytics Dashboard (Charts failed to load - <button onclick="refreshCharts()" style="background:none;border:none;color:var(--primary-color);text-decoration:underline;cursor:pointer;">Retry</button>)';
                    break;
                }
            }
        }
    }
    
    loadSampleGames() {
        console.log('Loading games by browsing API...');
        
        // Load popular games from different genres
        this.loadGamesByGenre('Action', 10);
        this.loadGamesByGenre('Adventure', 10);
        this.loadGamesByGenre('Indie', 10);
    }
    
    loadGamesByGenre(genre, limit) {
        if (!limit) limit = 20;
        
        console.log('Loading ' + genre + ' games...');
        
        fetch(this.apiBase + '/games/browse?genre=' + genre + '&limit=' + limit + '&rating_range=good')
            .then(response => {
                console.log(genre + ' games response status:', response.status);
                if (!response.ok) {
                    throw new Error('HTTP error! status: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                console.log(genre + ' games data received:', data);
                if (data.games && data.games.length > 0) {
                    this.renderGamesSection(data.games, genre + ' Games (' + data.games.length + ' games)', genre.toLowerCase());
                }
            })
            .catch(error => {
                console.error('Error loading ' + genre + ' games:', error);
            });
    }
    
    showGamesError(message) {
        const gamesSection = document.getElementById('gamesSection');
        if (gamesSection) {
            gamesSection.innerHTML = '<div class="error-message" style="color: red; padding: 20px; text-align: center;"><i class="fas fa-exclamation-triangle"></i> Failed to load games: ' + message + '<br><button onclick="dashboardInstance.loadSampleGames()" style="margin-top: 10px; padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Retry</button></div>';
        }
    }
    
    handleSearch() {
        const searchInput = document.getElementById('searchInput');
        const query = searchInput.value.trim();
        
        if (query.length < 2) {
            this.showError('Please enter at least 2 characters');
            return;
        }
        
        this.currentQuery = query;
        this.searchGames(query, 50);
    }
    
    searchGames(query, limit) {
        if (!limit) limit = 50;
        
        this.setLoading(true);
        
        fetch(this.apiBase + '/recommendations?query=' + encodeURIComponent(query) + '&limit=' + limit)
            .then(response => {
                if (!response.ok) {
                    throw new Error('HTTP error! status: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                const games = data.recommendations || data;
                const userProfile = data.user_profile || null;
                this.renderGames(games, 'Results for "' + query + '"', userProfile);
            })
            .catch(error => {
                console.error('Search error:', error);
                this.showError('Failed to search games. Please try again.');
            })
            .finally(() => {
                this.setLoading(false);
            });
    }
    
    renderStats(data) {
        const statsContainer = document.getElementById('statsContainer');
        if (!statsContainer) return;
        
        statsContainer.innerHTML = '<div class="stat-card"><h3><i class="fas fa-gamepad"></i> Total Games</h3><div class="stat-number">' + (data.total_games || 0).toLocaleString() + '</div></div><div class="stat-card"><h3><i class="fas fa-tags"></i> Genres</h3><div class="stat-number">' + Object.keys(data.genre_stats || {}).length + '</div></div><div class="stat-card"><h3><i class="fas fa-star"></i> Avg Rating</h3><div class="stat-number">85%</div></div><div class="stat-card"><h3><i class="fas fa-dollar-sign"></i> Free Games</h3><div class="stat-number">' + ((data.price_distribution && data.price_distribution['$0-5']) || 0).toLocaleString() + '</div></div>';
    }
    
    createCharts(data) {
        console.log('Creating charts with data:', data);
        
        if (!data) {
            console.error('No data provided for charts');
            return;
        }
        
        if (data.genre_stats) {
            this.createGenreChart(data.genre_stats);
        }
        
        if (data.price_distribution) {
            this.createPriceChart(data.price_distribution);
        }
        
        if (data.rating_distribution) {
            this.createRatingChart(data.rating_distribution);
        }
    }
    
    createGenreChart(genreStats) {
        const ctx = document.getElementById('genreChart');
        if (!ctx) return;
        
        const labels = Object.keys(genreStats).slice(0, 10);
        const values = Object.values(genreStats).slice(0, 10);
        
        if (this.charts.genre) {
            this.charts.genre.destroy();
        }
        
        this.charts.genre = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    createPriceChart(priceDistribution) {
        const ctx = document.getElementById('priceChart');
        if (!ctx) return;
        
        const labels = Object.keys(priceDistribution);
        const values = Object.values(priceDistribution);
        
        if (this.charts.price) {
            this.charts.price.destroy();
        }
        
        this.charts.price = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of Games',
                    data: values,
                    backgroundColor: '#36A2EB'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    createRatingChart(ratingDistribution) {
        const ctx = document.getElementById('ratingChart');
        if (!ctx) return;
        
        const labels = Object.keys(ratingDistribution);
        const values = Object.values(ratingDistribution);
        
        if (this.charts.rating) {
            this.charts.rating.destroy();
        }
        
        this.charts.rating = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: ['#FF6384', '#FFCE56', '#4BC0C0', '#36A2EB', '#9966FF']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    renderGamesSection(games, title, sectionId) {
        if (!sectionId) sectionId = 'main';
        
        console.log('Rendering games section:', title, 'Games:', games.length);
        
        // Create or find the games container
        let gamesContainer = document.getElementById('gamesContainer');
        if (!gamesContainer) {
            const mainSection = document.getElementById('gamesSection');
            if (mainSection) {
                mainSection.innerHTML = '<div id="gamesContainer"></div>';
                gamesContainer = document.getElementById('gamesContainer');
            }
        }
        
        if (!gamesContainer) {
            console.error('Games container not found');
            return;
        }
        
        // Create section HTML
        const sectionHtml = '<div class="games-section-wrapper" id="section-' + sectionId + '" style="margin-bottom: 2rem;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;"><h3><i class="fas fa-gamepad"></i> ' + title + '</h3><button onclick="dashboardInstance.loadMoreGames(\'' + sectionId + '\')" class="btn-secondary" style="padding: 0.5rem 1rem; background: #6c757d; color: white; border: none; border-radius: 6px; cursor: pointer;"><i class="fas fa-plus"></i> Load More</button></div><div class="games-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1.5rem;">' + this.renderGameCards(games) + '</div></div>';
        
        // Add section to container
        gamesContainer.insertAdjacentHTML('beforeend', sectionHtml);
    }
    
    renderGameCards(games) {
        return games.map(game => {
            const price = game.price === 0 ? 'FREE' : '$' + game.price.toFixed(2);
            const rating = (game.review_score * 100).toFixed(0) + '%';
            const reviews = game.total_reviews.toLocaleString();
            const genre = game.genres.split(',')[0];
            
            // Create fallback image if header_image is not available
            let imageHtml = '';
            if (game.header_image && game.header_image.trim() !== '' && game.header_image !== 'null') {
                // Clean the image URL and add proper error handling
                const cleanImageUrl = game.header_image.replace(/['"]/g, '');
                imageHtml = '<img src="' + cleanImageUrl + '" alt="' + game.name.replace(/['"]/g, '') + '" style="width: 100%; height: 100%; object-fit: cover; opacity: 0; transition: opacity 0.3s;" onload="this.style.opacity=1; console.log(\'Image loaded:\', this.src);" onerror="console.log(\'Image failed:\', this.src); this.style.display=\'none\'; this.parentElement.querySelector(\'.fallback-text\').style.display=\'flex\';">';
                imageHtml += '<div class="fallback-text" style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:white;font-size:0.9rem;text-align:center;padding:1rem;background:rgba(0,0,0,0.3);">' + game.name + '</div>';
            } else {
                imageHtml = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:white;font-size:0.9rem;text-align:center;padding:1rem;">' + game.name + '</div>';
            }
            
            return '<div class="game-card" style="background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.2s; cursor: pointer;" onclick="dashboardInstance.showGameDetails(' + game.app_id + ')" onmouseenter="this.style.transform=\'translateY(-4px)\'" onmouseleave="this.style.transform=\'translateY(0)\'"><div class="game-image" style="height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); position: relative; overflow: hidden;">' + imageHtml + '<div style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;">' + price + '</div></div><div style="padding: 1rem;"><h4 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #333; line-height: 1.3;">' + game.name + '</h4><div style="display: flex; align-items: center; margin-bottom: 0.5rem;"><div style="display: flex; align-items: center; margin-right: 1rem;"><i class="fas fa-star" style="color: #ffd700; margin-right: 4px;"></i><span style="font-size: 0.9rem; color: #666;">' + rating + '</span></div><div style="font-size: 0.8rem; color: #888;">' + reviews + ' reviews</div></div><div style="margin-bottom: 0.5rem;"><span style="font-size: 0.8rem; background: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 3px; margin-right: 4px;">' + genre + '</span></div><p style="margin: 0; font-size: 0.9rem; color: #666; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">' + game.description + '</p><div style="margin-top: 0.5rem; font-size: 0.8rem; color: #888;">' + game.developer + ' • ' + game.release_date + '</div></div></div>';
        }).join('');
    }
    
    renderGames(games, title, userProfile) {
        console.log('Rendering games (legacy method):', games, 'Title:', title, 'User Profile:', userProfile);
        this.renderGamesSection(games, title, 'legacy');
    }
    
    showGameDetails(appId) {
        console.log('Showing details for game:', appId);
        this.trackGameInteraction(appId, 'view');
        alert('Game details for ID: ' + appId + '\n\nThis would show detailed game information, screenshots, reviews, etc.');
    }
    
    trackGameInteraction(appId, interactionType) {
        console.log('Tracking interaction: ' + interactionType + ' for game ' + appId);
        
        fetch(this.apiBase + '/game/' + appId + '/interact', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                interaction_type: interactionType,
                timestamp: new Date().toISOString()
            })
        }).catch(error => console.log('Interaction tracking failed:', error));
    }
    
    loadMoreGames(sectionId) {
        console.log('Loading more games for section:', sectionId);
        
        // Find existing section and append more games
        const existingSection = document.getElementById('section-' + sectionId);
        if (existingSection) {
            const gamesGrid = existingSection.querySelector('.games-grid');
            if (gamesGrid) {
                // Load more games and append to existing grid
                this.loadAndAppendGames(sectionId, gamesGrid);
            }
        }
    }
    
    loadAndAppendGames(sectionId, gamesGrid) {
        let genre = '';
        if (sectionId === 'action') genre = 'Action';
        else if (sectionId === 'adventure') genre = 'Adventure';  
        else if (sectionId === 'indie') genre = 'Indie';
        
        if (!genre) return;
        
        fetch(this.apiBase + '/games/browse?genre=' + genre + '&limit=10&rating_range=good&page=2')
            .then(response => response.json())
            .then(data => {
                if (data.games && data.games.length > 0) {
                    gamesGrid.insertAdjacentHTML('beforeend', this.renderGameCards(data.games));
                }
            })
            .catch(error => {
                console.error('Error loading more games:', error);
            });
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        const loadingSection = document.getElementById('loadingSection');
        if (loadingSection) {
            loadingSection.style.display = loading ? 'block' : 'none';
        }
    }
    
    showError(message) {
        console.error('Dashboard error:', message);
        // You could show a toast notification or modal here
        alert('Error: ' + message);
    }
}

// Global dashboard instance
let dashboardInstance = null;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing dashboard...');
    console.log('Current URL:', window.location.href);
    console.log('API Base:', '/api');
    
    // Wait a bit more to ensure all scripts are loaded
    setTimeout(function() {
        console.log('Chart.js available:', typeof Chart !== 'undefined');
        console.log('Creating dashboard instance...');
        
        try {
            dashboardInstance = new GameMatchDashboard();
            window.dashboardInstance = dashboardInstance; // Make globally available
            console.log('Dashboard instance created successfully');
        } catch (error) {
            console.error('Error creating dashboard instance:', error);
        }
        
        // Fallback: Try to create charts again after a longer delay
        setTimeout(function() {
            if (typeof Chart !== 'undefined' && dashboardInstance) {
                console.log('Fallback: Attempting to create charts...');
                dashboardInstance.loadStats();
            } else {
                console.error('Dashboard instance or Chart.js not available for fallback');
            }
        }, 2000);
    }, 200);
});

// Add a manual refresh function for debugging
window.forceRefreshDashboard = function() {
    console.log('Force refreshing dashboard...');
    if (dashboardInstance) {
        dashboardInstance.loadStats();
        dashboardInstance.loadSampleGames();
    } else {
        console.log('Creating new dashboard instance...');
        dashboardInstance = new GameMatchDashboard();
    }
};

// Global function to refresh charts
function refreshCharts() {
    console.log('Manual chart refresh triggered');
    if (dashboardInstance) {
        console.log('Reloading stats and attempting chart creation...');
        dashboardInstance.loadStats();
        
        // Force chart creation after a delay
        setTimeout(function() {
            if (dashboardInstance.statsData) {
                console.log('Forcing chart creation with existing data...');
                dashboardInstance.createCharts(dashboardInstance.statsData);
            }
        }, 1000);
    } else {
        console.error('Dashboard instance not available');
    }
}

// Global function to test chart availability
window.testCharts = function() {
    console.log('Chart.js available:', typeof Chart !== 'undefined');
    console.log('Dashboard instance:', !!dashboardInstance);
    console.log('Stats data:', dashboardInstance ? !!dashboardInstance.statsData : 'No instance');
    
    if (dashboardInstance && dashboardInstance.statsData) {
        console.log('Attempting manual chart creation...');
        dashboardInstance.createCharts(dashboardInstance.statsData);
    }
};