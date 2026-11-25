const state = { sessionId: null, group: "", textbook: "", language: "en", currentTopic: "", nextTopic: "" };
let loadingNextQuestion = false;
let libraryByFilename = new Map();
let libraryReady = false;
let currentPlayerMode = null;
let currentPlayerResult = null;
let playerAnswerRevealed = false;
let currentLearningQuestion = null;
let lastScreeningResult = null;
let currentLearningLesson = null;
let graphEntityNames = [];
let selectedGraphEntity = "";
state.flow = "guided";
state.questionId = null;

function renderSuggestions(topic = null) {
  const current = topic || state.currentTopic;
  const suggestions = current ? [
    ...(state.nextTopic ? [`Ask from ${state.nextTopic}`] : []),
    `Change question`,
    `Summarize ${current} in 2 paragraphs`,
  ] : [
    `Summarize the main concepts in ${state.textbook || "this textbook"}`,
    `Explain a key topic from ${state.textbook || "this textbook"}`,
    `Change question`,
  ];
  $("#suggestions").replaceChildren(...suggestions.map((text) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => {
      if (text === "Change question") {
        changeQuestion();
      } else if (text.startsWith("Ask from ") && state.nextTopic) {
        askFromRelatedEntity();
      } else {
        $("#chat-input").value = text;
        submitChatMessage(text);
      }
    });
    return button;
  }));
}

async function changeQuestion() {
  if (loadingNextQuestion) return;
  $("#next-question-status").textContent = "Loading next question...";
  $("#next-question-status").classList.remove("hidden");
  try {
    await startLearningFlow();
    toast("Question changed!");
  } catch (err) {
    toast(err.message);
  }
}

async function askFromRelatedEntity() {
  if (!state.nextTopic || loadingNextQuestion) return;
  try {
    await startLearningFlow({ topic: state.nextTopic });
  } catch (err) {
    $("#next-question-status").classList.add("hidden");
    toast(err.message);
  }
}

async function sendMessage(message, topic = null) {
  if (!state.sessionId) await createSession();
  const response = await fetch(`/api/v1/sessions/${state.sessionId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      language: state.language,
      topic: topic || state.currentTopic || null,
    }),
  });
  if (!response.ok) throw new Error("No matching content was found in the selected textbook.");
  return response.json();
}

async function submitChatMessage(message) {
  const clean = (message || "").trim();
  if (!clean) return;
  const lower = clean.toLowerCase();
  const changeTriggers = ["change question", "next question", "another question", "new question", "skip question", "give me another question", "different question"];
  if (changeTriggers.includes(lower)) {
    $("#user-text").textContent = clean;
    $("#user-message").classList.remove("hidden");
    $("#assistant-text").textContent = "Loading a new question for you in the question player above...";
    $("#assistant-message").classList.remove("hidden");
    changeQuestion();
    return;
  }

  $("#user-text").textContent = clean;
  $("#user-message").classList.remove("hidden");
  $("#assistant-text").textContent = "Thinking...";
  $("#assistant-message").classList.remove("hidden");
  try {
    const result = await sendMessage(clean, state.currentTopic);
    $("#assistant-text").textContent = result.answer;
    $("#assistant-message").classList.remove("hidden");
    if (result.action === "change_question") {
      changeQuestion();
    }
    refreshProgress();
  } catch (error) {
    toast(error.message);
    $("#assistant-message").classList.add("hidden");
  }
}
const copy = {
  en: { studio: "Studio", placeholder: "Ask anything about your selected textbook..." },
  te: { studio: "స్టూడియో", placeholder: "మీరు ఎంచుకున్న పాఠ్యపుస్తకం గురించి అడగండి..." },
};
const $ = (selector) => document.querySelector(selector);
const displayName = (value) => value.length > 50 ? `${value.slice(0, 47)}...` : value;
const toast = (message) => {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  setTimeout(() => element.classList.remove("show"), 2600);
};

async function loadLibrary() {
  const response = await fetch("/api/v1/library");
  if (!response.ok) throw new Error("Could not load the indexed library.");
  const library = await response.json();
  libraryByFilename = new Map(library.textbooks.map((textbook) => [textbook.filename, textbook]));
  const textbooks = [...library.textbooks].sort((a, b) => {
    if (a.filename === "pib_prism.pdf") return -1;
    if (b.filename === "pib_prism.pdf") return 1;
    return 0;
  });
  const group = document.createElement("option");
  group.value = library.group;
  group.textContent = library.group;
  $("#group-select").replaceChildren(group);
  $("#textbook-select").replaceChildren(...textbooks.map((textbook) => {
    const option = document.createElement("option");
    option.value = textbook.filename;
    option.textContent = displayName(textbook.filename);
    return option;
  }));
  if (!library.textbooks.length) throw new Error("No indexed textbooks found. Ingest a PDF first.");
  libraryReady = true;
  $("#setup-submit").disabled = false;
}

function showLearningChoices() {
  const textbook = libraryByFilename.get(state.textbook);
  $("#selected-textbook").textContent = `Selected textbook: ${state.textbook}`;
  const topics = textbook?.topics || [];
  $("#flow-status").textContent = topics.length
    ? "Choose a textbook topic suggestion, enter your own topic, or continue with the textbook overview."
    : "No extracted topic suggestions are available yet. You can continue with the textbook overview.";
  $("#topic-suggestions").replaceChildren(...topics.map((topic) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = topic;
    button.addEventListener("click", () => { $("#flow-topic").value = topic; });
    return button;
  }));
}

async function createSession() {
  try {
    const response = await fetch("/api/v1/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: state.group, textbook: state.textbook }),
    });
    if (!response.ok) throw new Error("Session could not be created");
    state.sessionId = (await response.json()).session_id;
    $("#breadcrumb-text").textContent = displayName(state.textbook);
    $("#source-text").textContent = displayName(state.textbook);
    $("#source-status").textContent = "● Ready";
    $("#welcome-text").textContent = `Study ${state.textbook} using the indexed knowledge graph.`;
    renderSuggestions();
    refreshProgress();
    return true;
  } catch (error) {
    toast(error.message);
    return false;
  }
}

async function refreshProgress() {
  if (!state.sessionId) return;
  const response = await fetch(`/api/v1/sessions/${state.sessionId}/progress`);
  if (!response.ok) return;
  const progress = await response.json();
  $("#progress-bar").style.width = `${progress.percent}%`;
  $("#progress-percent").textContent = `${progress.percent}% explored`;
  $("#progress-label").textContent = `${progress.messages} messages · ${progress.questions} questions`;
}

async function startLearningFlow({ topic = null } = {}) {
  loadingNextQuestion = true;
  $("#next-question-status").textContent = "Loading the next question...";
  $("#next-question-status").classList.remove("hidden");
  let response;
  try {
    response = await fetch(`/api/v1/sessions/${state.sessionId}/learning-plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: state.flow,
        topic: topic || $("#flow-topic").value.trim(),
        language: state.language,
      }),
    });
  } catch (error) {
    loadingNextQuestion = false;
    $("#next-question-status").textContent = "The next question could not be loaded.";
    throw error;
  }
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      const error = JSON.parse(body);
      detail = error.detail || body;
    } catch (_) {
      if (!detail) detail = response.statusText;
    }
    loadingNextQuestion = false;
    throw new Error(detail || `Lesson request failed (HTTP ${response.status})`);
  }
  const plan = await response.json();
  state.questionId = plan.question.question_id;
  currentLearningQuestion = plan.question;
  currentLearningLesson = plan;
  lastScreeningResult = null;
  state.currentTopic = plan.topic;
  state.nextTopic = plan.next_topic || "";
  $("#lesson-topic").textContent = displayName(plan.topic);
  $("#lesson-source").textContent = plan.source === "fallback"
    ? "AI FALLBACK"
    : "FROM TEXTBOOK";
  $("#lesson-source").classList.toggle("fallback", plan.source === "fallback");
  renderLearningLesson();
  renderLearningQuestion();
  $("#screening-options").replaceChildren(...plan.question.options_en.map((option, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.option = String(index);
    button.textContent = `${String.fromCharCode(65 + index)}. ${option}`;
    button.addEventListener("click", () => submitScreening(index));
    return button;
  }));
  $("#attempts").textContent = "Two attempts allowed before the answer is revealed.";
  $("#next-question-status").classList.add("hidden");
  $("#screening-feedback").classList.add("hidden");
  $("#lesson").classList.remove("hidden");
  loadingNextQuestion = false;

  const askBtn = $("#ask-entity-btn");
  if (askBtn) {
    askBtn.classList.toggle("hidden", !state.nextTopic);
    if (state.nextTopic) {
      askBtn.textContent = `Ask from ${displayName(state.nextTopic)} ✦`;
    }
  }
  renderSuggestions(plan.topic);

}

function renderLearningQuestion() {
  if (!currentLearningQuestion) return;
  const hasTeluguQuestion = Boolean(
    currentLearningQuestion.question_te
    && currentLearningQuestion.options_te
    && currentLearningQuestion.options_te.length === currentLearningQuestion.options_en?.length
  );
  const question = state.language === "te" && hasTeluguQuestion
    ? currentLearningQuestion.question_te
    : currentLearningQuestion.question_en || currentLearningQuestion.question;
  const options = state.language === "te" && hasTeluguQuestion
    ? currentLearningQuestion.options_te
    : currentLearningQuestion.options_en || currentLearningQuestion.options;
  $("#screening-question").textContent = question;
  $("#screening-options").querySelectorAll("[data-option]").forEach((button, index) => {
    button.textContent = `${String.fromCharCode(65 + index)}. ${options?.[index] || ""}`;
  });
  if (lastScreeningResult && !$("#screening-feedback").classList.contains("hidden")) {
    $("#screening-feedback").textContent = state.language === "te"
      ? lastScreeningResult.feedback_te
      : lastScreeningResult.feedback_en;
  }
}

function renderLearningLesson() {
  if (!currentLearningLesson) return;
  const useTelugu = state.language === "te";
  $("#lesson-etext").textContent = useTelugu
    ? (currentLearningLesson.etext_te || currentLearningLesson.etext_en || currentLearningLesson.etext)
    : (currentLearningLesson.etext_en || currentLearningLesson.etext);
  $("#lesson-summary").textContent = useTelugu
    ? (currentLearningLesson.summary_te || currentLearningLesson.summary_en || currentLearningLesson.summary)
    : (currentLearningLesson.summary_en || currentLearningLesson.summary);
}

async function submitScreening(optionIndex) {
  if (loadingNextQuestion) return;
  document.querySelectorAll("[data-option]").forEach((button) => { button.disabled = true; });
  $("#next-question-status").textContent = "Checking your answer...";
  $("#next-question-status").classList.remove("hidden");
  let response;
  try {
    response = await fetch(`/api/v1/sessions/${state.sessionId}/screening/${state.questionId}/attempt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ option_index: optionIndex, language: state.language }),
    });
  } catch (error) {
    document.querySelectorAll("[data-option]").forEach((button) => { button.disabled = false; });
    $("#next-question-status").classList.add("hidden");
    toast(error.message);
    return;
  }
  if (!response.ok) {
    document.querySelectorAll("[data-option]").forEach((button) => { button.disabled = false; });
    $("#next-question-status").classList.add("hidden");
    toast("Could not record this attempt.");
    return;
  }
  const result = await response.json();
  lastScreeningResult = result;
  const selected = document.querySelector(`[data-option="${optionIndex}"]`);
  selected.classList.add(result.correct ? "correct" : "incorrect");
  $("#attempts").textContent = `${result.attempts_remaining} attempt(s) remaining`;
  $("#screening-feedback").textContent = state.language === "te"
    ? result.feedback_te
    : result.feedback_en;
  $("#screening-feedback").classList.remove("hidden");
  if (result.answer_revealed) {
    document.querySelector(`[data-option="${result.correct_option_index}"]`).classList.add("correct");
    $("#next-question-status").textContent = "Moving on to the next question...";
    $("#next-question-status").classList.remove("hidden");
    setTimeout(() => {
      startLearningFlow().catch((error) => toast(error.message));
    }, 700);
  } else {
    document.querySelectorAll("[data-option]").forEach((button) => {
      button.disabled = button.classList.contains("incorrect");
    });
    $("#next-question-status").classList.add("hidden");
  }
  refreshProgress();
}

$("#change-question-btn")?.addEventListener("click", () => changeQuestion());
$("#ask-entity-btn")?.addEventListener("click", () => askFromRelatedEntity());

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  await submitChatMessage(message);
});

$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!state.sessionId) await createSession();
    try {
      await loadPlayerQuestion(button.dataset.mode);
    } catch (error) {
      toast(error.message);
    }
  });
});

async function loadPlayerQuestion(mode) {
  const response = await fetch(`/api/v1/sessions/${state.sessionId}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, language: state.language }),
  });
  if (!response.ok) {
    let detail = "Could not load a question.";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch (error) {
      console.warn("Question request returned a non-JSON error", error);
    }
    throw new Error(detail);
  }

  const result = await response.json();
  currentPlayerMode = mode;
  currentPlayerResult = result;
  playerAnswerRevealed = false;
  renderPlayerQuestion(result);
}

function renderPlayerQuestion(sourceResult) {
  const hasTeluguQuestion = Boolean(
    sourceResult.question_te
    && sourceResult.options_te
    && sourceResult.options_te.length === (sourceResult.options_en || sourceResult.options || []).length
  );
  const result = {
    ...sourceResult,
    question: state.language === "te"
      ? (hasTeluguQuestion ? sourceResult.question_te : sourceResult.question_en || sourceResult.question)
      : sourceResult.question_en || sourceResult.question,
    options: state.language === "te"
      ? (hasTeluguQuestion ? sourceResult.options_te : sourceResult.options_en || sourceResult.options)
      : sourceResult.options_en || sourceResult.options,
    answer: state.language === "te"
      ? (hasTeluguQuestion ? sourceResult.answer_te || sourceResult.answer : sourceResult.answer_en || sourceResult.answer)
      : sourceResult.answer_en || sourceResult.answer,
  };
  $("#user-text").textContent = `Play a ${result.mode} question`;
  $("#user-message").classList.remove("hidden");
  $("#assistant-text").textContent = result.question;
  document.querySelectorAll(".player-options, .player-answer, .player-like, .player-log")
    .forEach((item) => item.remove());

  const options = document.createElement("div");
  options.className = "player-options";
  result.options.forEach((value, index) => {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "player-option";
    option.dataset.playerOption = String(index);
    option.textContent = `${String.fromCharCode(65 + index)}. ${value}`;
    option.addEventListener("click", () => {
      options.querySelectorAll("button").forEach((item) => { item.disabled = true; });
      answer.textContent = `${state.language === "te" ? "సమాధానం" : "Answer"}: ${result.answer}`;
      answer.classList.remove("hidden");
      playerAnswerRevealed = true;
    });
    options.append(option);
  });
  const answer = document.createElement("div");
  answer.className = playerAnswerRevealed ? "player-answer" : "player-answer hidden";
  if (playerAnswerRevealed) {
    answer.textContent = `${state.language === "te" ? "సమాధానం" : "Answer"}: ${result.answer}`;
  }
  const like = document.createElement("button");
  like.type = "button";
  like.className = "outline-button player-like";
  like.setAttribute("aria-pressed", String(result.liked));
  like.textContent = result.liked ? "♥ Liked" : "♡ Like question";
  like.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const liked = button.getAttribute("aria-pressed") !== "true";
    try {
      const likeResponse = await fetch(
        `/api/v1/sessions/${state.sessionId}/questions/${result.question_id}/like`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ liked }),
        },
      );
      if (!likeResponse.ok) throw new Error("Could not save your preference.");
      button.setAttribute("aria-pressed", String(liked));
      button.textContent = liked ? "♥ Liked" : "♡ Like question";
      document.querySelector(".player-log").textContent = liked
        ? "Saved as a liked fallback question."
        : "Removed from liked fallback questions.";
    } catch (error) {
      toast(error.message);
    }
  });
  const log = document.createElement("small");
  log.className = "player-log";
  log.textContent = `Question ready from ${result.liked ? "liked-question fallback" : "question player"}.`;
  $("#assistant-text").after(options, answer, like, log);
  $("#assistant-message").classList.remove("hidden");
  refreshProgress();
  $("#chat").scrollIntoView({ behavior: "smooth" });
}

$("#new-session").addEventListener("click", async () => {
  await createSession();
  currentPlayerMode = null;
  currentPlayerResult = null;
  playerAnswerRevealed = false;
  $("#user-message").classList.add("hidden");
  $("#assistant-message").classList.add("hidden");
  toast("New study space ready");
});

$("#mobile-menu").addEventListener("click", () => {
  document.querySelector(".sidebar").classList.toggle("open");
});

document.querySelectorAll(".sidebar a, .sidebar button").forEach((element) => {
  element.addEventListener("click", () => {
    document.querySelector(".sidebar").classList.remove("open");
  });
});

$("#setup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!libraryReady) {
    toast("Loading your indexed textbooks. Please select one when it appears.");
    return;
  }
  state.group = $("#group-select").value;
  state.textbook = $("#textbook-select").value;
  const created = await createSession();
  if (!created) return;
  $("#setup-overlay").remove();
  showLearningChoices();
  $("#flow-overlay").classList.remove("hidden");
});

$("#open-graph").addEventListener("click", async () => {
  if (!state.textbook) {
    toast("Select a textbook before opening its knowledge graph.");
    return;
  }
  selectedGraphEntity = "";
  graphEntityNames = [];
  $("#graph-entity-search").value = "";
  $("#graph-entities").replaceChildren();
  $("#play-graph-question").disabled = true;
  $("#graph-selected").textContent = "";
  $("#graph-selected").classList.add("hidden");
  $("#graph-status").textContent = "Loading entities...";
  $("#graph-overlay").classList.remove("hidden");

  try {
    const response = await fetch(`/api/v1/graph?document_id=${encodeURIComponent(state.textbook)}`);
    if (!response.ok) throw new Error("Knowledge graph is unavailable.");
    const graph = await response.json();
    const ignoredLabels = new Set([
      "Document", "Chunk", "StudyGroup", "Textbook", "StudySession",
      "StudyMessage", "ScreeningQuestion", "StudyQuestion",
    ]);
    const byName = new Map();
    for (const node of graph.nodes || []) {
      const name = String(node.name || "").trim();
      const labels = Array.isArray(node.labels) ? node.labels : [];
      if (
        !name
        || name.length > 140
        || /^[0-9a-f]{32,}$/i.test(name)
        || labels.some((label) => ignoredLabels.has(label))
      ) continue;
      if (!byName.has(name.toLocaleLowerCase())) {
        byName.set(name.toLocaleLowerCase(), {
          name,
          type: labels[0] || "Entity",
        });
      }
    }
    graphEntityNames = [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
    renderGraphEntities();
    $("#graph-status").textContent = graphEntityNames.length
      ? `${graphEntityNames.length} entities found in ${state.textbook}. Select one to continue.`
      : `No selectable entities were found in ${state.textbook}.`;
  } catch (error) {
    $("#graph-status").textContent = error.message;
  }
});

function renderGraphEntities() {
  const query = $("#graph-entity-search").value.trim().toLocaleLowerCase();
  const entities = graphEntityNames.filter((entity) => entity.name.toLocaleLowerCase().includes(query));
  $("#graph-entities").replaceChildren(...entities.map((entity) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "graph-entity";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(entity.name === selectedGraphEntity));
    const name = document.createElement("span");
    name.className = "graph-entity-name";
    name.textContent = entity.name;
    const type = document.createElement("span");
    type.className = "graph-entity-type";
    type.textContent = entity.type;
    button.append(name, type);
    button.addEventListener("click", () => {
      selectedGraphEntity = entity.name;
      $("#play-graph-question").disabled = false;
      $("#graph-selected").textContent = `Selected entity: ${entity.name}`;
      $("#graph-selected").classList.remove("hidden");
      renderGraphEntities();
    });
    return button;
  }));
  if (query) {
    $("#graph-status").textContent = entities.length
      ? `Showing ${entities.length} matching entities.`
      : "No entities match your search.";
  } else if (graphEntityNames.length) {
    $("#graph-status").textContent = `${graphEntityNames.length} entities found in ${state.textbook}. Select one to continue.`;
  }
}

$("#graph-entity-search").addEventListener("input", renderGraphEntities);
$("#close-graph").addEventListener("click", () => {
  $("#graph-overlay").classList.add("hidden");
});
$("#graph-overlay").addEventListener("click", (event) => {
  if (event.target === $("#graph-overlay")) $("#graph-overlay").classList.add("hidden");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("#graph-overlay").classList.add("hidden");
});
$("#play-graph-question").addEventListener("click", async () => {
  if (!selectedGraphEntity || loadingNextQuestion) return;
  const button = $("#play-graph-question");
  button.disabled = true;
  try {
    await startLearningFlow({ topic: selectedGraphEntity });
    $("#graph-overlay").classList.add("hidden");
    $("#lesson").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = !selectedGraphEntity;
  }
});

loadLibrary().catch((error) => toast(error.message));

document.querySelectorAll("[data-flow]").forEach((button) => {
  button.addEventListener("click", () => {
    state.flow = button.dataset.flow;
    document.querySelectorAll("[data-flow]").forEach((item) => item.classList.toggle("active-flow", item === button));
  });
});

$("#flow-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await startLearningFlow();
    $("#flow-overlay").classList.add("hidden");
    window.scrollTo(0, 0);
  } catch (error) {
    toast(error.message);
  }
});

const originalTextNodes = new WeakMap();
const originalAttributes = new WeakMap();
const teluguTextCache = new Map();
const originalPageTitle = document.title;
let translationQueue = Promise.resolve();
let mutatingForTranslation = false;

function shouldTranslateText(text) {
  const value = text.trim();
  if (!value || value.length > 400) return false;
  if (/^[\d\s.%()+\-*/:;,.!?'"“”_×→←↻•●♥♡✦▣▤◌⌕↑↓]+$/.test(value)) return false;
  if (/[\u0C00-\u0C7F]/.test(value)) return false;
  return true;
}

function collectTranslatableNodes(root = document.body) {
  const nodes = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  let node = walker.nextNode();
  while (node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const parent = node.parentElement;
      if (
        parent
        && !parent.closest("script, style, [data-no-translate]")
        && shouldTranslateText(node.nodeValue)
      ) {
        nodes.push(node);
      }
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      for (const attribute of ["placeholder", "title", "aria-label"]) {
        const value = node.getAttribute(attribute);
        if (value && shouldTranslateText(value)) {
          nodes.push({ element: node, attribute, value });
        }
      }
    }
    node = walker.nextNode();
  }
  return nodes;
}

function rememberOriginalText(node) {
  if (!originalTextNodes.has(node)) {
    originalTextNodes.set(node, node.nodeValue);
  }
  return originalTextNodes.get(node);
}

function rememberOriginalAttribute(element, attribute) {
  let values = originalAttributes.get(element);
  if (!values) {
    values = {};
    originalAttributes.set(element, values);
  }
  if (!(attribute in values)) values[attribute] = element.getAttribute(attribute);
  return values[attribute];
}

const staticTeluguCopy = new Map([
  ["WELCOME TO LEARNING STUDIO", "లెర్నింగ్ స్టూడియోకు స్వాగతం"],
  ["Set up your study space", "మీ అధ్యయన స్థలాన్ని సెటప్ చేయండి"],
  ["Choose your group and textbook. The tutor will use your library first, then Gemini when your graph has no answer.", "మీ గ్రూప్ మరియు పాఠ్యపుస్తకాన్ని ఎంచుకోండి. మీ గ్రాఫ్కు సమాధానం లేనప్పుడు ట్యూటర్ ముందుగా మీ లైబ్రరీని, తర్వాత Geminiని ఉపయోగిస్తుంది."],
  ["Group", "గ్రూప్"],
  ["Textbook", "పాఠ్యపుస్తకం"],
  ["Start learning", "అధ్యయనం ప్రారంభించండి"],
  ["YOUR NEXT STEP", "మీ తదుపరి దశ"],
  ["How do you want to learn?", "మీరు ఎలా నేర్చుకోవాలనుకుంటున్నారు?"],
  ["Guided study", "మార్గదర్శక అధ్యయనం"],
  ["Build from fundamentals with a recommended sequence.", "సిఫార్సు చేసిన క్రమంలో పునాదుల నుండి నేర్చుకోండి."],
  ["Topic focused", "విషయ కేంద్రిత"],
  ["Jump straight into one topic you choose.", "మీరు ఎంచుకున్న ఒక విషయానికి నేరుగా వెళ్లండి."],
  ["Choose a topic", "ఒక విషయాన్ని ఎంచుకోండి"],
  ["Choose a suggestion or continue without one", "సూచనను ఎంచుకోండి లేదా దానితో లేకుండా కొనసాగండి"],
  ["Suggestions are taken from the selected textbook. You can choose one or continue with the textbook overview.", "సూచనలు ఎంచుకున్న పాఠ్యపుస్తకం నుండి తీసుకోబడ్డాయి. మీరు ఒకదాన్ని ఎంచుకోవచ్చు లేదా పాఠ్యపుస్తక అవలోకనంతో కొనసాగవచ్చు."],
  ["Start this lesson", "ఈ పాఠాన్ని ప్రారంభించండి"],
  ["Learning Studio", "లెర్నింగ్ స్టూడియో"],
  ["your thinking space", "మీ ఆలోచనల స్థలం"],
  ["New study space", "కొత్త అధ్యయన స్థలం"],
  ["AI tutor", "AI ట్యూటర్"],
  ["Question player", "ప్రశ్న ప్లేయర్"],
  ["My library", "నా లైబ్రరీ"],
  ["Knowledge graph", "నాలెడ్జ్ గ్రాఫ్"],
  ["Recent spaces", "ఇటీవలి స్థలాలు"],
  ["Settings", "సెట్టింగ్‌లు"],
  ["Personal workspace", "వ్యక్తిగత వర్క్‌స్పేస్"],
  ["My library", "నా లైబ్రరీ"],
  ["Share", "షేర్"],
  ["AI TUTOR", "AI ట్యూటర్"],
  ["What are you working on today?", "ఈ రోజు మీరు దేని మీద పనిచేస్తున్నారు?"],
  ["Ask questions, explore your documents, or let me quiz you on what you know.", "ప్రశ్నలు అడగండి, మీ పత్రాలను అన్వేషించండి లేదా మీకు తెలిసిన దానిపై నేను క్విజ్ చేయనివ్వండి."],
  ["No textbook selected", "పాఠ్యపుస్తకం ఎంచుకోలేదు"],
  ["Studio", "స్టూడియో"],
  ["You", "మీరు"],
  ["just now", "ఇప్పుడే"],
  ["now", "ఇప్పుడు"],
  ["Ask anything about your study material...", "మీ అధ్యయన సామగ్రి గురించి ఏదైనా అడగండి..."],
  ["↵ to send · shift + ↵ for new line", "పంపడానికి ↵ · కొత్త పంక్తికి shift + ↵"],
  ["Studio answers from your sources and shows its working.", "స్టూడియో మీ మూలాల నుంచే సమాధానం ఇస్తుంది మరియు పనిచేసిన విధానాన్ని చూపిస్తుంది."],
  ["LEARNING STEP", "అధ్యయన దశ"],
  ["IN SHORT", "సంక్షిప్తంగా"],
  ["KNOWLEDGE CHECK", "జ్ఞాన పరీక్ష"],
  ["Change question", "ప్రశ్నను మార్చండి"],
  ["Ask from topic", "విషయం ఆధారంగా అడగండి"],
  ["Two attempts allowed before the answer is revealed.", "సమాధానం వెల్లడయ్యే ముందు రెండు ప్రయత్నాలకు అవకాశం ఉంది."],
  ["Checking your answer...", "మీ సమాధానాన్ని పరిశీలిస్తున్నాం..."],
  ["Moving on to the next question...", "తదుపరి ప్రశ్నకు వెళ్తున్నాం..."],
  ["Loading the next question...", "తదుపరి ప్రశ్నను లోడ్ చేస్తున్నాం..."],
  ["Thinking...", "ఆలోచిస్తున్నాం..."],
  ["STUDY TOOLS", "అధ్యయన సాధనాలు"],
  ["Make it stick", "దీన్ని బాగా గుర్తుంచుకోండి"],
  ["Turn this chapter into active recall.", "ఈ అధ్యాయాన్ని చురుకైన జ్ఞాపకశక్తి సాధనగా మార్చండి."],
  ["From textbook", "పాఠ్యపుస్తకం నుండి"],
  ["Previous questions", "మునుపటి ప్రశ్నలు"],
  ["Repeat what I missed", "నాకు తప్పిన వాటిని పునరావృతం చేయండి"],
  ["Study progress", "అధ్యయన పురోగతి"],
  ["Keep exploring your material", "మీ అధ్యయన సామగ్రిని అన్వేషిస్తూ ఉండండి"],
  ["0% explored", "0% అన్వేషించబడింది"],
  ["See how ideas connect.", "ఆలోచనలు ఎలా అనుసంధానమవుతాయో చూడండి."],
  ["Open graph", "గ్రాఫ్ తెరవండి"],
  ["KNOWLEDGE GRAPH", "నాలెడ్జ్ గ్రాఫ్"],
  ["Choose an entity", "ఒక ఎంటిటీని ఎంచుకోండి"],
  ["Select a textbook entity, then play a question about it.", "పాఠ్యపుస్తకంలోని ఎంటిటీని ఎంచుకుని, దాని గురించి ప్రశ్నను ప్రారంభించండి."],
  ["Search entities", "ఎంటిటీలను వెతకండి"],
  ["Type an entity name", "ఎంటిటీ పేరును టైప్ చేయండి"],
  ["Play question", "ప్రశ్నను ప్రారంభించండి"],
  ["Close knowledge graph", "నాలెడ్జ్ గ్రాఫ్‌ను మూసివేయండి"],
  ["TODAY'S FOCUS", "ఈ రోజు దృష్టి"],
  ["The goal of learning is not to memorize, but to understand the relationships.", "నేర్చుకోవడం లక్ష్యం కేవలం గుర్తుపట్టడం కాదు, సంబంధాలను అర్థం చేసుకోవడం."],
  ["Based on your recent questions", "మీ ఇటీవలి ప్రశ్నల ఆధారంగా"],
  ["English selected", "ఇంగ్లీష్ ఎంచుకోబడింది"],
  ["తెలుగు ఎంచుకోబడింది", "తెలుగు ఎంచుకోబడింది"],
]);

async function translateStrings(texts) {
  const unique = [...new Set(texts)];
  unique.forEach((text) => {
    if (staticTeluguCopy.has(text.trim())) {
      teluguTextCache.set(text, staticTeluguCopy.get(text.trim()));
    }
  });
  const missing = unique.filter((text) => !teluguTextCache.has(text));
  for (let index = 0; index < missing.length; index += 30) {
    const batch = missing.slice(index, index + 30);
    let translated = false;
    for (let attempt = 0; attempt < 2 && !translated; attempt += 1) {
      try {
        const response = await fetch("/api/v1/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: batch, target_language: "te" }),
        });
        if (!response.ok) throw new Error("Telugu translation could not be loaded.");
        const result = await response.json();
        if (!Array.isArray(result.translations) || result.translations.length !== batch.length) {
          throw new Error("Telugu translation returned an incomplete result.");
        }
        batch.forEach((text, position) => {
          teluguTextCache.set(text, result.translations[position] || text);
        });
        translated = true;
      } catch (error) {
        if (attempt === 1) toast(error.message);
        else await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }
  }
  return texts.map((text) => teluguTextCache.get(text) || text);
}

async function translatePageToTelugu() {
  mutatingForTranslation = true;
  translationQueue = translationQueue
    .catch(() => {})
    .then(async () => {
      const items = collectTranslatableNodes();
      const textNodes = items.filter((item) => item.nodeType === Node.TEXT_NODE);
      const attributes = items.filter((item) => item.element);
      const strings = [
        originalPageTitle,
        ...textNodes.map((node) => rememberOriginalText(node).trim()),
        ...attributes.map((item) => rememberOriginalAttribute(item.element, item.attribute).trim()),
      ];
      if (!strings.length) return;
      const translations = await translateStrings(strings);
      document.title = translations[0];
      textNodes.forEach((node, index) => {
        const original = rememberOriginalText(node);
        const leading = original.match(/^(\s*)/)?.[1] || "";
        const trailing = original.match(/(\s*)$/)?.[1] || "";
        node.nodeValue = `${leading}${translations[index + 1]}${trailing}`;
      });
      attributes.forEach((item, offset) => {
        item.element.setAttribute(
          item.attribute,
          translations[textNodes.length + offset + 1],
        );
      });
    })
    .finally(() => {
      mutatingForTranslation = false;
    });
  await translationQueue;
}

function restorePageToEnglish() {
  document.title = originalPageTitle;
  for (const node of document.querySelectorAll("*")) {
    const values = originalAttributes.get(node);
    if (values) {
      for (const [attribute, value] of Object.entries(values)) {
        if (value !== null) node.setAttribute(attribute, value);
        else node.removeAttribute(attribute);
      }
    }
  }
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    if (originalTextNodes.has(node)) node.nodeValue = originalTextNodes.get(node);
    node = walker.nextNode();
  }
  document.querySelectorAll("[data-copy]").forEach((element) => {
    element.textContent = copy.en[element.dataset.copy];
  });
  $("#chat-input").placeholder = copy.en.placeholder;
}

const pageMutationObserver = new MutationObserver((mutations) => {
  if (!mutations.length || mutatingForTranslation || state.language !== "te") return;
  const changedNodes = mutations.flatMap((mutation) => {
    if (mutation.type === "characterData") {
      return mutation.target.parentElement ? [mutation.target.parentElement] : [];
    }
    return mutation.addedNodes.length ? [...mutation.addedNodes] : [];
  });
  for (const node of changedNodes) {
    if (node.nodeType === Node.ELEMENT_NODE) {
      translateStringsForRoot(node);
    }
  }
});

async function translateStringsForRoot(root) {
  mutatingForTranslation = true;
  await translationQueue.catch(() => {});
  const items = collectTranslatableNodes(root);
  const textNodes = items.filter((item) => item.nodeType === Node.TEXT_NODE);
  const attributes = items.filter((item) => item.element);
  const strings = [
    ...textNodes.map((node) => rememberOriginalText(node).trim()),
    ...attributes.map((item) => rememberOriginalAttribute(item.element, item.attribute).trim()),
  ];
  if (!strings.length) {
    mutatingForTranslation = false;
    return;
  }
  const translations = await translateStrings(strings);
  textNodes.forEach((node, index) => {
    node.nodeValue = translations[index];
  });
  attributes.forEach((item, offset) => {
    item.element.setAttribute(item.attribute, translations[textNodes.length + offset]);
  });
  mutatingForTranslation = false;
}

pageMutationObserver.observe(document.body, {
  childList: true,
  characterData: true,
  subtree: true,
});

$("#language-switch").addEventListener("click", async () => {
  state.language = state.language === "en" ? "te" : "en";
  document.documentElement.lang = state.language;
  document.querySelector(".lang-en").classList.toggle("active-lang", state.language === "en");
  document.querySelector(".lang-te").classList.toggle("active-lang", state.language === "te");
  document.querySelectorAll(".setup-lang-en, .flow-lang-en").forEach((element) => {
    element.classList.toggle("active-lang", state.language === "en");
  });
  document.querySelectorAll(".setup-lang-te, .flow-lang-te").forEach((element) => {
    element.classList.toggle("active-lang", state.language === "te");
  });
  if (state.language === "te") {
    await translatePageToTelugu();
  } else {
    restorePageToEnglish();
  }
  renderLearningQuestion();
  renderLearningLesson();
  if (currentPlayerMode && currentPlayerResult) {
    renderPlayerQuestion(currentPlayerResult);
  }
  toast(state.language === "te" ? "తెలుగు ఎంచుకోబడింది" : "English selected");
});

document.querySelectorAll("#setup-language-switch, #flow-language-switch").forEach((button) => {
  button.addEventListener("click", () => $("#language-switch").click());
});
