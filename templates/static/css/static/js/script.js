document.addEventListener('DOMContentLoaded', function() {
    const detectionForm = document.getElementById('detectionForm');
    const textInput = document.getElementById('textInput');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsCard = document.getElementById('resultsCard');
    const resultBadge = document.getElementById('resultBadge');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceScore = document.getElementById('confidenceScore');
    const highlightedText = document.getElementById('highlightedText');
    const totalSentences = document.getElementById('totalSentences');
    const avgSentenceLength = document.getElementById('avgSentenceLength');
    const vocabRichness = document.getElementById('vocabRichness');
    const comparisonTableBody = document.getElementById('comparisonTableBody');
    
    detectionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const text = textInput.value.trim();
        
        if (text.length < 50) {
            alert('Please enter at least 50 characters for accurate analysis.');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        resultsCard.style.display = 'none';
        
        // Send data to backend
        fetch('/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.style.display = 'none';
            alert('An error occurred during analysis. Please try again.');
        });
    });
    
    function displayResults(data) {
        // Show results card
        resultsCard.style.display = 'block';
        
        // Set result badge
        resultBadge.textContent = data.result;
        resultBadge.className = 'result-badge ' + data.result.toLowerCase().replace(' ', '-');
        
        // Set confidence bar
        const percentage = Math.round(data.ai_probability * 100);
        confidenceFill.style.width = percentage + '%';
        confidenceFill.className = 'confidence-fill ' + data.result.toLowerCase().replace(' ', '-');
        confidenceScore.textContent = percentage + '%';
        
        // Set highlighted text
        highlightedText.innerHTML = data.highlighted_text;
        
        // Add tooltips to highlighted spans
        const spans = highlightedText.querySelectorAll('span[data-score]');
        spans.forEach(span => {
            const score = parseFloat(span.getAttribute('data-score'));
            const percentage = Math.round(score * 100);
            span.title = `AI Probability: ${percentage}%`;
            
            // Add hover effect
            span.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.02)';
                this.style.transition = 'transform 0.2s';
            });
            
            span.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        });
        
        // Set detailed analysis
        if (data.detailed_analysis && data.detailed_analysis.statistics) {
            totalSentences.textContent = data.detailed_analysis.statistics.total_sentences;
            
            // Calculate average sentence length
            const words = textInput.value.trim().split(/\s+/);
            const avgLength = Math.round(words.length / data.detailed_analysis.statistics.total_sentences);
            avgSentenceLength.textContent = avgLength + ' words';
            
            // Set vocabulary richness
            const uniqueWords = new Set(words.map(word => word.toLowerCase()));
            const richness = Math.round((uniqueWords.size / words.length) * 100);
            vocabRichness.textContent = richness + '%';
        }
        
        // Set comparison table
        if (data.comparative_analysis && data.comparative_analysis.scores) {
            comparisonTableBody.innerHTML = '';
            
            const scores = data.comparative_analysis.scores;
            const toolNames = {
                'lemadh_ai': 'Lemadh AI',
                'originality_ai': 'Originality.AI',
                'copyleaks': 'Copyleaks',
                'winston_ai': 'Winston AI',
                'zerogpt': 'ZeroGPT'
            };
            
            for (const [tool, score] of Object.entries(scores)) {
                const row = document.createElement('tr');
                const percentage = Math.round(score * 100);
                
                row.innerHTML = `
                    <td class="tool-name">${toolNames[tool]}</td>
                    <td class="score ${tool === 'lemadh_ai' ? 'best-score' : ''}">${percentage}%</td>
                `;
                
                comparisonTableBody.appendChild(row);
            }
        }
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
