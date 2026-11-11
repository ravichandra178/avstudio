const state = { sessionId: null, group: "", textbook: "", language: "en", currentTopic: "" };
let loadingNextQuestion = false;
let libraryByFilename = new Map();
let libraryReady = false;
state.flow = "guided";
state.questionId = null;

function renderSuggestions(topic = null) {
  const current = topic || state.currentTopic;
  const suggestions = current ? [
    `Ask from ${current}`,
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
  const group = document.createElement("option");
  group.value = library.group;
  group.textContent = library.group;
  $("#group-select").replaceChildren(group);
  $("#textbook-select").replaceChildren(...library.textbooks.map((textbook) => {
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

async function startLearningFlow() {
  loadingNextQuestion = true;
  $("#next-question-status").textContent = "Loading the next question...";
  $("#next-question-status").classList.remove("hidden");
  let response;
  try {
    response = await fetch(`/api/v1/sessions/${state.sessionId}/learning-plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: state.flow, topic: $("#flow-topic").value.trim(), language: state.language }),
    });
  } catch (error) {
    loadingNextQuestion = false;
    $("#next-question-status").textContent = "The next question could not be loaded.";
    throw error;
  }
  if (!response.ok) {
    let detail = "The textbook lesson could not be created.";
    try { detail = (await response.json()).detail || detail; } catch (_) { /* use default */ }
    loadingNextQuestion = false;
    throw new Error(detail);
  }
  const plan = await response.json();
  state.questionId = plan.question.question_id;
  state.currentTopic = plan.topic;
  $("#lesson-topic").textContent = displayName(plan.topic);
  $("#lesson-source").textContent = plan.source === "fallback"
    ? "AI FALLBACK"
    : "FROM TEXTBOOK";
  $("#lesson-source").classList.toggle("fallback", plan.source === "fallback");
  $("#lesson-etext").textContent = plan.etext;
  $("#lesson-summary").textContent = plan.summary;
  $("#screening-question").textContent = plan.question.question;
  $("#screening-options").innerHTML = plan.question.options.map((option, index) => `<button type="button" data-option="${index}">${option}</button>`).join("");
  $("#attempts").textContent = "Two attempts allowed before the answer is revealed.";
  $("#next-question-status").classList.add("hidden");
  $("#screening-feedback").classList.add("hidden");
  $("#lesson").classList.remove("hidden");
  loadingNextQuestion = false;

  const askBtn = $("#ask-entity-btn");
  if (askBtn) {
    askBtn.textContent = `Ask from ${displayName(plan.topic)} ✦`;
  }
  renderSuggestions(plan.topic);

  document.querySelectorAll("[data-option]").forEach((button) => button.addEventListener("click", () => submitScreening(Number(button.dataset.option))));
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
      body: JSON.stringify({ option_index: optionIndex }),
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
  const selected = document.querySelector(`[data-option="${optionIndex}"]`);
  selected.classList.add(result.correct ? "correct" : "incorrect");
  $("#attempts").textContent = `${result.attempts_remaining} attempt(s) remaining`;
  $("#screening-feedback").textContent = result.feedback;
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
$("#ask-entity-btn")?.addEventListener("click", () => {
  if (state.currentTopic) {
    const prompt = `Ask from ${state.currentTopic}`;
    $("#chat-input").value = prompt;
    submitChatMessage(prompt);
  }
});

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
      const response = await fetch(`/api/v1/sessions/${state.sessionId}/questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: button.dataset.mode, language: state.language }),
      });
      if (!response.ok) throw new Error("Add a textbook or previous question first.");
      const result = await response.json();
      $("#user-text").textContent = `Play a ${result.mode} question`;
      $("#user-message").classList.remove("hidden");
      $("#assistant-text").textContent = result.question;
      const options = result.options.map((option, index) => {
        return `<button type="button" class="player-option" data-player-option="${index}">${String.fromCharCode(65 + index)}. ${option}</button>`;
      }).join("");
      document.querySelectorAll(".player-options, .player-answer").forEach((item) => item.remove());
      $("#assistant-text").insertAdjacentHTML(
        "afterend",
        `<div class="player-options">${options}</div><div class="player-answer hidden"></div>
         <button type="button" class="outline-button player-like" aria-pressed="${result.liked}">
           ${result.liked ? "♥ Liked" : "♡ Like question"}
         </button><small class="player-log">Question ready from ${result.liked ? "liked-question fallback" : "question player"}.</small>`,
      );
      document.querySelectorAll("[data-player-option]").forEach((option) => {
        option.addEventListener("click", () => {
          document.querySelectorAll("[data-player-option]").forEach((item) => { item.disabled = true; });
          const answer = document.querySelector(".player-answer");
          answer.textContent = `Answer: ${result.answer}`;
          answer.classList.remove("hidden");
        });
        document.querySelector(".player-like").addEventListener("click", async (event) => {
          const button = event.currentTarget;
          const liked = button.getAttribute("aria-pressed") !== "true";
          try {
            const response = await fetch(
              `/api/v1/sessions/${state.sessionId}/questions/${result.question_id}/like`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ liked }),
              },
            );
            if (!response.ok) throw new Error("Could not save your preference.");
            button.setAttribute("aria-pressed", String(liked));
            button.textContent = liked ? "♥ Liked" : "♡ Like question";
            document.querySelector(".player-log").textContent = liked
              ? "Saved as a liked fallback question."
              : "Removed from liked fallback questions.";
          } catch (error) {
            toast(error.message);
          }
        });
      });
      $("#assistant-message").classList.remove("hidden");
      refreshProgress();
      $("#chat").scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      toast(error.message);
    }
  });
});

$("#new-session").addEventListener("click", async () => {
  await createSession();
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
  const response = await fetch(`/api/v1/graph?document_id=${encodeURIComponent(state.textbook)}`);
  if (!response.ok) { toast("Knowledge graph is unavailable."); return; }
  const graph = await response.json();
  $("#user-text").textContent = "Open knowledge graph";
  $("#user-message").classList.remove("hidden");
  $("#assistant-text").textContent = `${graph.nodes.length} nodes and ${graph.relationships.length} relationships indexed for ${state.textbook}.`;
  $("#citation").classList.add("hidden");
  $("#assistant-message").classList.remove("hidden");
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
  } catch (error) {
    toast(error.message);
  }
});

$("#language-switch").addEventListener("click", () => {
  state.language = state.language === "en" ? "te" : "en";
  document.querySelector(".lang-en").classList.toggle("active-lang", state.language === "en");
  document.querySelector(".lang-te").classList.toggle("active-lang", state.language === "te");
  document.querySelectorAll("[data-copy]").forEach((element) => {
    element.textContent = copy[state.language][element.dataset.copy];
  });
  $("#chat-input").placeholder = copy[state.language].placeholder;
  toast(state.language === "te" ? "తెలుగు ఎంచుకోబడింది" : "English selected");
});
