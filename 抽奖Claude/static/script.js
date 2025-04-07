// script.js
document.addEventListener('DOMContentLoaded', function() {
    // 元素引用
    const drawButton = document.getElementById('draw-btn');
    const nameRoll = document.getElementById('name-roll');
    const winnersDisplay = document.getElementById('winners');
    const currentPrize = document.getElementById('current-prize');
    const remainingCount = document.getElementById('remaining-count');
    const prizeNameInput = document.getElementById('prize-name');
    const winnerCountInput = document.getElementById('winner-count');
    const historyList = document.getElementById('history-list');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const closeModal = document.getElementById('close-modal');
    const candidatesList = document.getElementById('candidates-list');
    const saveCandidatesBtn = document.getElementById('save-candidates');
    const resetWinnersBtn = document.getElementById('reset-winners');
    
    // 标签切换
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // 移除所有标签的active类
            tabs.forEach(t => t.classList.remove('active'));
            // 添加当前标签的active类
            this.classList.add('active');
            
            // 隐藏所有标签内容
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            
            // 显示当前标签内容
            const tabName = this.getAttribute('data-tab');
            document.getElementById(`${tabName}-pane`).classList.add('active');
        });
    });
    
    // 抽奖状态
    let isDrawing = false;
    let candidates = [];
    let remainingCandidates = [];
    let drawInterval;
    let rollSpeed = 50; // ms
    
    // 初始化
    fetchCandidates();
    fetchHistory();
    
    // 获取候选人列表
    function fetchCandidates() {
        fetch('/api/candidates')
            .then(response => response.json())
            .then(data => {
                candidates = data.candidates;
                remainingCandidates = data.remaining;
                updateRemainingCount();
                
                // 填充候选人文本框
                candidatesList.value = candidates.join('\n');
            })
            .catch(error => console.error('获取候选人失败:', error));
    }
    
    // 获取历史记录
    function fetchHistory() {
        fetch('/api/history')
            .then(response => response.json())
            .then(data => {
                renderHistory(data.history);
            })
            .catch(error => console.error('获取历史记录失败:', error));
    }
    
    // 渲染历史记录
    function renderHistory(history) {
        if (!history || history.length === 0) {
            historyList.innerHTML = '<div class="loading">暂无记录</div>';
            return;
        }
        
        historyList.innerHTML = '';
        // 倒序显示历史记录
        history.slice().reverse().forEach(record => {
            const item = document.createElement('div');
            item.className = 'history-item';
            
            const prize = document.createElement('div');
            prize.className = 'history-prize';
            prize.textContent = record.prize;
            
            const winners = document.createElement('div');
            winners.className = 'history-winners';
            
            record.winners.forEach(winner => {
                const span = document.createElement('span');
                span.className = 'history-winner';
                span.textContent = winner;
                winners.appendChild(span);
            });
            
            const time = document.createElement('div');
            time.className = 'history-time';
            time.textContent = record.timestamp;
            
            item.appendChild(prize);
            item.appendChild(winners);
            item.appendChild(time);
            
            historyList.appendChild(item);
        });
    }
    
    // 更新剩余人数显示
    function updateRemainingCount() {
        remainingCount.textContent = `剩余参与者: ${remainingCandidates.length} 人`;
    }
    
    // 开始/停止抽奖
    drawButton.addEventListener('click', function() {
        if (isDrawing) {
            stopDraw();
        } else {
            startDraw();
        }
    });
    
    // 开始抽奖动画
    function startDraw() {
        if (remainingCandidates.length === 0) {
            alert('没有可抽取的候选人了！');
            return;
        }
        
        const count = parseInt(winnerCountInput.value) || 1;
        if (count > remainingCandidates.length) {
            alert(`只剩下 ${remainingCandidates.length} 人可抽取，已自动调整`);
            winnerCountInput.value = remainingCandidates.length;
        }
        
        isDrawing = true;
        drawButton.textContent = '停止抽奖';
        drawButton.classList.add('warning-btn');
        currentPrize.textContent = prizeNameInput.value || '奖项';
        winnersDisplay.innerHTML = '';
        
        // 开始动画
        drawInterval = setInterval(() => {
            const randomIndex = Math.floor(Math.random() * remainingCandidates.length);
            const randomName = remainingCandidates[randomIndex];
            nameRoll.textContent = randomName;
        }, rollSpeed);
    }
    
    // 停止抽奖并显示结果
    function stopDraw() {
        if (!isDrawing) return;
        
        clearInterval(drawInterval);
        isDrawing = false;
        drawButton.textContent = '开始抽奖';
        drawButton.classList.remove('warning-btn');
        
        // 抽取获奖者
        const prizeName = prizeNameInput.value || '奖项';
        const count = parseInt(winnerCountInput.value) || 1;
        
        fetch('/api/draw', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prizeName: prizeName,
                count: count
            })
        })
        .then(response => response.json())
        .then(data => {
            // 清空滚动区域
            nameRoll.textContent = '';
            
            // 显示获奖者
            winnersDisplay.innerHTML = '';
            data.winners.forEach(winner => {
                const div = document.createElement('div');
                div.className = 'winner-item';
                div.textContent = winner;
                winnersDisplay.appendChild(div);
            });
            
            // 刷新数据
            fetchCandidates();
            fetchHistory();
        })
        .catch(error => {
            console.error('抽奖出错:', error);
            alert('抽奖过程中出现错误，请重试！');
            nameRoll.textContent = '出错了';
        });
    }
    
    // 设置按钮
    settingsBtn.addEventListener('click', function() {
        settingsModal.style.display = 'block';
    });
    
    // 关闭弹窗
    closeModal.addEventListener('click', function() {
        settingsModal.style.display = 'none';
    });
    
    // 点击弹窗外部关闭
    window.addEventListener('click', function(event) {
        if (event.target === settingsModal) {
            settingsModal.style.display = 'none';
        }
    });
    
    // 保存候选人名单
    saveCandidatesBtn.addEventListener('click', function() {
        const text = candidatesList.value.trim();
        if (!text) {
            alert('请输入候选人名单');
            return;
        }
        
        const newCandidates = text.split('\n')
            .map(name => name.trim())
            .filter(name => name); // 过滤空行
        
        if (newCandidates.length === 0) {
            alert('请输入有效的候选人名单');
            return;
        }
        
        fetch('/api/candidates', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                candidates: newCandidates
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('候选人名单保存成功！');
                settingsModal.style.display = 'none';
                fetchCandidates();
                fetchHistory();
            }
        })
        .catch(error => {
            console.error('保存名单失败:', error);
            alert('保存名单失败，请重试！');
        });
    });
    
    // 重置中奖记录
    resetWinnersBtn.addEventListener('click', function() {
        if (confirm('确定要重置所有中奖记录吗？所有人将重新参与抽奖。')) {
            fetch('/api/reset', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('中奖记录已重置！');
                    fetchCandidates();
                    fetchHistory();
                }
            })
            .catch(error => {
                console.error('重置失败:', error);
                alert('重置失败，请重试！');
            });
        }
    });
});