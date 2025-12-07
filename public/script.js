const chatbox = document.getElementById("chatbox");
const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");

sendBtn.onclick = sendMessage;
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

function appendMessage(sender, text) {
  const bubble = document.createElement("div");
  bubble.className = sender === "user"
    ? "bg-blue-600 text-white p-3 rounded-lg mb-2 self-end max-w-[80%] ml-auto"
    : "bg-gray-200 text-gray-900 p-3 rounded-lg mb-2 max-w-[80%]";

  // escape HTML to prevent XSS
  let safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // convert newlines to <br>
  safe = safe.replace(/\n/g, "<br>");

  // pretty bullet points
  safe = safe.replace(/•/g, "<span class='ml-4'>•</span>");

  bubble.innerHTML = safe;
  chatbox.appendChild(bubble);
  chatbox.scrollTop = chatbox.scrollHeight;
}


async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage("user", text);
  userInput.value = "";

  // Loading bubble
  const loading = document.createElement("div");
  loading.className = "bg-gray-300 text-gray-700 p-3 rounded-lg mb-2 max-w-[80%]";
  loading.textContent = "MediAssist is thinking...";
  chatbox.appendChild(loading);
  chatbox.scrollTop = chatbox.scrollHeight;

  try {
    const res = await fetch("http://127.0.0.1:8080/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text })
    });

    const data = await res.json();
    loading.remove();
    appendMessage("bot", data.answer);

  } catch (err) {
    loading.remove();
    appendMessage("bot", "Error: Cannot connect to the server.");
  }
}
