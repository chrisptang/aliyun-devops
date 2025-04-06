document.addEventListener('DOMContentLoaded', function() {
    const nameDisplay = document.getElementById('name-display');
    const prizeDisplay = document.getElementById('prize-display');
    const startBtn = document.getElementById('start-btn');
    const prizeNameInput = document.getElementById('prize-name');
    const winnerCountInput = document.getElementById('winner-count');
    const remainingCount = document.getElementById('remaining-count');
    const candidatesText = document.getElementById('candidates-text');
    const loadDefaultBtn = document.getElementById('load-default-btn');
    const updateBtn = document.getElementById('update-btn');
    const winnersList = document.getElementById('winners-list');

    let isRunning = false;
    let animationInterval;
    let candidates = [];
    let winners = [];

    // Initialize the app
    fetchCandidates();
    loadDefaultCandidates();

    // Event listeners
    startBtn.addEventListener('click', toggleLottery);
    loadDefaultBtn.addEventListener('click', loadDefaultCandidates);
    updateBtn.addEventListener('click', updateCandidates);

    // Toggle lottery start/stop
    function toggleLottery() {
        if (isRunning) {
            stopLottery();
        } else {
            startLottery();
        }
    }

    // Start the lottery animation
    function startLottery() {
        const prizeName = prizeNameInput.value.trim();
        if (!prizeName) {
            alert('请输入奖项名称');
            return;
        }

        isRunning = true;
        startBtn.textContent = '停止抽奖';
        prizeDisplay.textContent = prizeName;

        // Animation effect
        animationInterval = setInterval(() => {
            const randomIndex = Math.floor(Math.random() * candidates.length);
            nameDisplay.textContent = candidates[randomIndex];
        }, 100);
    }

    // Stop the lottery and get winners
    function stopLottery() {
        clearInterval(animationInterval);
        isRunning = false;
        startBtn.textContent = '开始抽奖';

        const prizeName = prizeNameInput.value.trim();
        const winnerCount = parseInt(winnerCountInput.value);

        // Call API to get winners
        fetch('/start_lottery', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prize_name: prizeName,
                winner_count: winnerCount
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            // Display winners
            nameDisplay.textContent = data.winners.join('、');
            nameDisplay.classList.add('highlight');
            
            // Update UI
            remainingCount.textContent = data.remaining_candidates;
            addWinnersToDisplay(data.winners, prizeName);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Add winners to the display list
    function addWinnersToDisplay(winnerNames, prizeName) {
        winnerNames.forEach(name => {
            const winnerItem = document.createElement('div');
            winnerItem.className = 'winner-item';
            winnerItem.innerHTML = `
                <span>${name}</span>
                <span>${prizeName}</span>
            `;
            winnersList.prepend(winnerItem);
        });
    }

    // Fetch current candidates from server
    function fetchCandidates() {
        fetch('/get_candidates')
        .then(response => response.json())
        .then(data => {
            candidates = data.candidates;
            remainingCount.textContent = data.count;
            candidatesText.value = candidates.join('\n');
        });
    }

    // Load default candidates
    function loadDefaultCandidates() {
        fetch('/get_candidates')
        .then(response => response.json())
        .then(data => {
            candidatesText.value = data.candidates.join('\n');
        });
    }

    // Update candidates list
    function updateCandidates() {
        const newCandidates = candidatesText.value.split('\n')
            .map(name => name.trim())
            .filter(name => name.length > 0);

        // In a real app, we would send this to the server to update
        candidates = newCandidates;
        remainingCount.textContent = candidates.length;
    }
});
