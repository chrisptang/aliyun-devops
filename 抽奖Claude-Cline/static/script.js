// 全局变量
let lotteryData = {
    candidates: [],
    prizes: [],
    history: []
};
let isDrawing = false;
let animationInterval;
let currentPrizeId = null;

// DOM 元素
const elements = {
    // 标签页
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // 抽奖界面
    prizeSelect: document.getElementById('prize-select'),
    lotteryAnimation: document.getElementById('lottery-animation'),
    nameDisplay: document.querySelector('.name-display'),
    winnerDisplay: document.getElementById('winner-display'),
    currentPrize: document.getElementById('current-prize'),
    remainingCount: document.getElementById('remaining-count'),
    candidateCount: document.getElementById('candidate-count'),
    startBtn: document.getElementById('start-btn'),
    stopBtn: document.getElementById('stop-btn'),
    resetBtn: document.getElementById('reset-btn'),
    
    // 设置界面
    prizesList: document.getElementById('prizes-list'),
    addPrizeBtn: document.getElementById('add-prize-btn'),
    savePrizesBtn: document.getElementById('save-prizes-btn'),
    candidatesInput: document.getElementById('candidates-input'),
    saveCandidatesBtn: document.getElementById('save-candidates-btn'),
    resetCandidatesBtn: document.getElementById('reset-candidates-btn'),
    
    // 历史记录界面
    historyList: document.getElementById('history-list'),
    
    // 模板
    prizeItemTemplate: document.getElementById('prize-item-template'),
    historyItemTemplate: document.getElementById('history-item-template'),
    winnerItemTemplate: document.getElementById('winner-item-template')
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 标签页切换
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
    
    // 抽奖控制
    elements.startBtn.addEventListener('click', startLottery);
    elements.stopBtn.addEventListener('click', stopLottery);
    elements.resetBtn.addEventListener('click', resetLottery);
    elements.prizeSelect.addEventListener('change', updatePrizeInfo);
    
    // 设置控制
    elements.addPrizeBtn.addEventListener('click', addPrizeItem);
    elements.savePrizesBtn.addEventListener('click', savePrizes);
    elements.saveCandidatesBtn.addEventListener('click', saveCandidates);
    elements.resetCandidatesBtn.addEventListener('click', resetCandidates);
    
    // 加载数据
    fetchData();
});

// 切换标签页
function switchTab(tabId) {
    elements.tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    
    elements.tabContents.forEach(content => {
        content.classList.toggle('active', content.id === tabId);
    });
}

// 从服务器获取数据
async function fetchData() {
    try {
        const response = await fetch('/api/data');
        lotteryData = await response.json();
        
        // 更新界面
        updatePrizeSelect();
        updatePrizeInfo();
        updateCandidatesInput();
        updatePrizesList();
        updateHistoryList();
    } catch (error) {
        console.error('获取数据失败:', error);
        showNotification('获取数据失败，请刷新页面重试', 'error');
    }
}

// 更新奖项下拉选择框
function updatePrizeSelect() {
    elements.prizeSelect.innerHTML = '';
    
    lotteryData.prizes.forEach(prize => {
        const option = document.createElement('option');
        option.value = prize.name;
        option.textContent = `${prize.name} (${prize.winners.length}/${prize.count})`;
        elements.prizeSelect.appendChild(option);
    });
    
    if (elements.prizeSelect.options.length > 0) {
        elements.prizeSelect.selectedIndex = 0;
        currentPrizeId = elements.prizeSelect.value;
    }
}

// 更新奖项信息显示
function updatePrizeInfo() {
    if (!elements.prizeSelect.value) return;
    
    currentPrizeId = elements.prizeSelect.value;
    const prize = lotteryData.prizes.find(p => p.name === currentPrizeId);
    
    if (prize) {
        elements.currentPrize.textContent = prize.name;
        elements.remainingCount.textContent = `${prize.count - prize.winners.length}/${prize.count}`;
        
        // 计算可用候选人数量
        const allWinners = getAllWinners();
        const availableCandidates = lotteryData.candidates.filter(c => !allWinners.includes(c));
        elements.candidateCount.textContent = availableCandidates.length;
        
        // 禁用开始按钮条件：已抽完或没有可用候选人
        elements.startBtn.disabled = (prize.winners.length >= prize.count) || (availableCandidates.length === 0);
    }
}

// 获取所有已中奖人员
function getAllWinners() {
    const winners = [];
    lotteryData.prizes.forEach(prize => {
        winners.push(...prize.winners);
    });
    return winners;
}

// 更新候选人输入框
function updateCandidatesInput() {
    elements.candidatesInput.value = lotteryData.candidates.join('\n');
}

// 更新奖项列表
function updatePrizesList() {
    elements.prizesList.innerHTML = '';
    
    lotteryData.prizes.forEach(prize => {
        const prizeItem = createPrizeItem(prize.name, prize.count);
        elements.prizesList.appendChild(prizeItem);
    });
}

// 创建奖项设置项
function createPrizeItem(name = '', count = 1) {
    const template = elements.prizeItemTemplate.content.cloneNode(true);
    const prizeItem = template.querySelector('.prize-item');
    
    const nameInput = prizeItem.querySelector('.prize-name');
    const countInput = prizeItem.querySelector('.prize-count');
    const removeBtn = prizeItem.querySelector('.remove-prize-btn');
    
    nameInput.value = name;
    countInput.value = count;
    
    removeBtn.addEventListener('click', () => {
        prizeItem.remove();
    });
    
    return prizeItem;
}

// 添加奖项设置项
function addPrizeItem() {
    const prizeItem = createPrizeItem();
    elements.prizesList.appendChild(prizeItem);
}

// 保存奖项设置
async function savePrizes() {
    const prizeItems = elements.prizesList.querySelectorAll('.prize-item');
    const prizes = [];
    
    prizeItems.forEach(item => {
        const name = item.querySelector('.prize-name').value.trim();
        const count = parseInt(item.querySelector('.prize-count').value);
        
        if (name && count > 0) {
            // 查找现有奖项以保留已有的获奖者
            const existingPrize = lotteryData.prizes.find(p => p.name === name);
            const winners = existingPrize ? existingPrize.winners : [];
            
            prizes.push({
                name,
                count,
                winners
            });
        }
    });
    
    if (prizes.length === 0) {
        showNotification('请至少添加一个有效的奖项', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/prizes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prizes })
        });
        
        const result = await response.json();
        
        if (result.success) {
            lotteryData.prizes = result.prizes;
            updatePrizeSelect();
            updatePrizeInfo();
            showNotification('奖项设置已保存', 'success');
        } else {
            showNotification('保存奖项设置失败', 'error');
        }
    } catch (error) {
        console.error('保存奖项设置失败:', error);
        showNotification('保存奖项设置失败，请重试', 'error');
    }
}

// 保存候选人设置
async function saveCandidates() {
    const candidatesText = elements.candidatesInput.value.trim();
    let candidates = [];
    
    if (candidatesText) {
        candidates = candidatesText.split('\n')
            .map(name => name.trim())
            .filter(name => name.length > 0);
    }
    
    if (candidates.length === 0) {
        showNotification('请至少添加一个候选人', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/candidates', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ candidates })
        });
        
        const result = await response.json();
        
        if (result.success) {
            lotteryData.candidates = result.candidates;
            updatePrizeInfo();
            showNotification('候选人设置已保存', 'success');
        } else {
            showNotification('保存候选人设置失败', 'error');
        }
    } catch (error) {
        console.error('保存候选人设置失败:', error);
        showNotification('保存候选人设置失败，请重试', 'error');
    }
}

// 重置候选人为默认名单
function resetCandidates() {
    fetchData();
    showNotification('已恢复默认候选人名单', 'success');
}

// 更新历史记录列表
function updateHistoryList() {
    elements.historyList.innerHTML = '';
    
    if (lotteryData.history.length === 0) {
        const emptyMessage = document.createElement('div');
        emptyMessage.className = 'empty-message';
        emptyMessage.textContent = '暂无抽奖记录';
        elements.historyList.appendChild(emptyMessage);
        return;
    }
    
    lotteryData.history.forEach(record => {
        const template = elements.historyItemTemplate.content.cloneNode(true);
        const historyItem = template.querySelector('.history-item');
        
        historyItem.querySelector('.history-prize').textContent = record.prize;
        historyItem.querySelector('.history-winner').textContent = record.winner;
        historyItem.querySelector('.history-time').textContent = record.timestamp;
        
        elements.historyList.appendChild(historyItem);
    });
}

// 开始抽奖
function startLottery() {
    if (isDrawing) return;
    
    const prize = lotteryData.prizes.find(p => p.name === currentPrizeId);
    if (!prize) return;
    
    // 检查是否已抽完
    if (prize.winners.length >= prize.count) {
        showNotification(`${prize.name}已抽完`, 'error');
        return;
    }
    
    // 获取可用候选人
    const allWinners = getAllWinners();
    const availableCandidates = lotteryData.candidates.filter(c => !allWinners.includes(c));
    
    if (availableCandidates.length === 0) {
        showNotification('没有可用的候选人', 'error');
        return;
    }
    
    isDrawing = true;
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;
    elements.lotteryAnimation.classList.add('highlight');
    
    // 开始动画
    let lastIndex = -1;
    animationInterval = setInterval(() => {
        let randomIndex;
        do {
            randomIndex = Math.floor(Math.random() * availableCandidates.length);
        } while (randomIndex === lastIndex && availableCandidates.length > 1);
        
        lastIndex = randomIndex;
        elements.nameDisplay.textContent = availableCandidates[randomIndex];
        elements.nameDisplay.style.transform = `scale(${0.9 + Math.random() * 0.2})`;
    }, 100);
}

// 停止抽奖
async function stopLottery() {
    if (!isDrawing) return;
    
    clearInterval(animationInterval);
    elements.nameDisplay.style.transform = 'scale(1)';
    elements.lotteryAnimation.classList.remove('highlight');
    
    try {
        const response = await fetch('/api/draw', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prizeId: currentPrizeId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // 显示中奖动画
            elements.lotteryAnimation.classList.add('winner-animation');
            elements.nameDisplay.textContent = result.winner;
            
            setTimeout(() => {
                elements.lotteryAnimation.classList.remove('winner-animation');
                
                // 添加到获奖展示区
                const template = elements.winnerItemTemplate.content.cloneNode(true);
                const winnerItem = template.querySelector('.winner-item');
                
                winnerItem.querySelector('.winner-name').textContent = result.winner;
                winnerItem.querySelector('.winner-prize').textContent = result.prize;
                
                elements.winnerDisplay.appendChild(winnerItem);
                
                // 更新数据
                fetchData();
            }, 3000);
            
            showNotification(`恭喜 ${result.winner} 获得 ${result.prize}！`, 'success');
        } else {
            showNotification(`抽奖失败: ${result.message}`, 'error');
        }
    } catch (error) {
        console.error('抽奖请求失败:', error);
        showNotification('抽奖请求失败，请重试', 'error');
    }
    
    isDrawing = false;
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
}

// 重置抽奖
async function resetLottery() {
    if (!confirm('确定要重置抽奖吗？这将清空所有中奖记录，但保留奖项和候选人设置。')) {
        return;
    }
    
    try {
        const response = await fetch('/api/reset', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            elements.winnerDisplay.innerHTML = '';
            fetchData();
            showNotification('抽奖已重置', 'success');
        } else {
            showNotification('重置抽奖失败', 'error');
        }
    } catch (error) {
        console.error('重置抽奖失败:', error);
        showNotification('重置抽奖失败，请重试', 'error');
    }
}

// 显示通知
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // 添加到页面
    document.body.appendChild(notification);
    
    // 显示动画
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // 自动关闭
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// 添加通知样式
const notificationStyle = document.createElement('style');
notificationStyle.textContent = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px 20px;
    border-radius: 5px;
    color: white;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 1000;
    transform: translateX(120%);
    transition: transform 0.3s ease;
}

.notification.show {
    transform: translateX(0);
}

.notification.success {
    background-color: var(--success-color);
}

.notification.error {
    background-color: var(--danger-color);
}

.notification.info {
    background-color: #17a2b8;
}
`;
document.head.appendChild(notificationStyle);
