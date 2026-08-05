const form = document.querySelector('#generate-form');
const promptInput = document.querySelector('#prompt');
const charCount = document.querySelector('#char-count');
const player = document.querySelector('#player');
const emptyPreview = document.querySelector('#empty-preview');
const choiceOverlay = document.querySelector('#choice-overlay');
const progress = document.querySelector('#progress');
const progressBar = document.querySelector('#progress-bar');
const progressPercent = document.querySelector('#progress-percent');
const progressTitle = document.querySelector('#progress-title');
const progressDetail = document.querySelector('#progress-detail');
const resultSection = document.querySelector('#story-result');
const previewActions = document.querySelector('#preview-actions');
let generated = null;
let progressTimer = null;

promptInput.addEventListener('input', () => {
  charCount.textContent = `${promptInput.value.length} / 300`;
});
promptInput.dispatchEvent(new Event('input'));

function beginProgress() {
  const phases = [
    [12, '스토리를 구성하고 있어요', '등장인물과 장면을 설계하는 중…'],
    [30, '캐릭터의 목소리를 만들고 있어요', '장면별 한국어 TTS를 생성하는 중…'],
    [52, '세로형 장면을 연출하고 있어요', '9:16 이미지와 강조 자막을 그리는 중…'],
    [74, 'Wan2.2가 장면 영상을 만들고 있어요', '장면 이미지와 동작 프롬프트를 클라우드에서 처리 중…'],
    [91, '두 개의 결말을 연결하고 있어요', 'Wan2.2 영상, 음성과 자막을 MP4로 마무리하는 중…'],
  ];
  let index = 0;
  progress.classList.remove('hidden');
  const update = () => {
    const [percent, title, detail] = phases[Math.min(index, phases.length - 1)];
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    progressTitle.textContent = title;
    progressDetail.textContent = detail;
    index += 1;
  };
  update();
  progressTimer = setInterval(update, 4500);
}

function finishProgress() {
  clearInterval(progressTimer);
  progressBar.style.width = '100%';
  progressPercent.textContent = '100%';
  progressTitle.textContent = '드라마가 완성됐어요';
  progressDetail.textContent = '재생 버튼을 눌러 이야기를 확인해보세요.';
  setTimeout(() => progress.classList.add('hidden'), 1600);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  beginProgress();

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: promptInput.value.trim(),
        genre: document.querySelector('#genre').value,
        mood: document.querySelector('#mood').value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '영상 생성에 실패했습니다.');
    generated = payload;
    renderResult(payload);
    finishProgress();
  } catch (error) {
    clearInterval(progressTimer);
    progress.classList.remove('hidden');
    progressTitle.textContent = '생성하지 못했어요';
    progressDetail.textContent = error.message;
    progressPercent.textContent = '!';
  } finally {
    button.disabled = false;
  }
});

function renderResult(data) {
  emptyPreview.classList.add('hidden');
  player.style.display = 'block';
  player.loop = false;
  player.muted = false;
  player.poster = data.poster;
  player.src = data.videos.common;
  player.load();
  document.querySelector('#preview-status').textContent = '공통편 40초 · 결말을 선택하세요';

  document.querySelector('#story-title').textContent = data.story.title;
  document.querySelector('#story-logline').textContent = data.story.logline;
  const allScenes = [...data.story.common_scenes, ...data.story.ending_a, ...data.story.ending_b];
  document.querySelector('#scene-list').innerHTML = allScenes.map((scene, index) => `
    <article class="scene-card" style="--scene-color:${scene.accent}">
      <div class="scene-meta"><span>SCENE ${String(index + 1).padStart(2, '0')}</span><span>${scene.duration} SEC</span></div>
      <h3>${escapeHtml(scene.title)}</h3>
      <p><span class="scene-speaker">${escapeHtml(scene.speaker)}</span>${escapeHtml(scene.dialogue)}</p>
    </article>
  `).join('');
  resultSection.classList.remove('hidden');
  previewActions.classList.remove('hidden');
  document.querySelector('#download-a').href = data.videos.full_a;
  document.querySelector('#download-a').download = `${data.story.title}_결말A.mp4`;
  document.querySelector('#download-b').href = data.videos.full_b;
  document.querySelector('#download-b').download = `${data.story.title}_결말B.mp4`;
  const endingButtons = choiceOverlay.querySelectorAll('button');
  endingButtons[0].textContent = `A. ${data.story.ending_a[0].title}`;
  endingButtons[1].textContent = `B. ${data.story.ending_b[0].title}`;
  resultSection.scrollIntoView({behavior: 'smooth', block: 'start'});
}

player.addEventListener('ended', () => {
  if (generated && player.src.includes('common.mp4')) choiceOverlay.classList.remove('hidden');
});

choiceOverlay.querySelectorAll('button').forEach((button) => {
  button.addEventListener('click', () => {
    const ending = button.dataset.ending;
    choiceOverlay.classList.add('hidden');
    player.src = generated.videos[`ending_${ending}`];
    player.play();
    document.querySelector('#preview-status').textContent = `결말 ${ending.toUpperCase()} 재생 중`;
  });
});

document.querySelector('#replay-common').addEventListener('click', () => {
  if (!generated) return;
  choiceOverlay.classList.add('hidden');
  player.src = generated.videos.common;
  player.play();
  document.querySelector('#preview-status').textContent = '공통편 40초 · 결말을 선택하세요';
});

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[character]);
}
