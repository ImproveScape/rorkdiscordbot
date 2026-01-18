import os
import re
import asyncio
import discord
from discord import app_commands
from discord.ext import commands
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCS_URL = os.getenv("DOCS_URL", "https://docs.rork.com")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@dataclass
class WebChunk:
    url: str
    title: str
    section: str
    content: str
    embedding: List[float] = None

class WebKnowledgeBase:
    def __init__(self):
        self.chunks: List[WebChunk] = []
        self.is_indexing = False
        self.index_progress = ""
        self.pages_count = 0

    def clear(self):
        self.chunks = []
        self.pages_count = 0

    def add_chunk(self, chunk: WebChunk):
        self.chunks.append(chunk)

    def search(self, query_embedding: List[float], top_k: int = 6) -> List[WebChunk]:
        if not self.chunks:
            return []
        scored = []
        for chunk in self.chunks:
            if chunk.embedding:
                score = sum(a * b for a, b in zip(query_embedding, chunk.embedding))
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

kb = WebKnowledgeBase()

def get_embedding(text: str) -> List[float]:
    text = text[:8000]
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

class DocsScraper:
    def __init__(self, base_url: str, max_pages: int = 300):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visited: Set[str] = set()
        self.pages = []

    def normalize_url(self, url: str) -> str:
        url = url.split('#')[0].rstrip('/')
        if not url.startswith('http'):
            url = urljoin(self.base_url, url)
        return url

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False
        skip = ['/assets/', '/static/', '/images/', '/_next/', '.png', '.jpg', '.gif', '.svg', '.css', '.js', '.json']
        return not any(p in url.lower() for p in skip)

    async def scrape_page(self, session, url):
        url = self.normalize_url(url)
        if url in self.visited:
            return None
        self.visited.add(url)
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200 or 'text/html' not in resp.headers.get('content-type', ''):
                    return None
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                title = ""
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)
                elif soup.title:
                    title = soup.title.string or ""
                main = soup.find('main') or soup.find('article') or soup.body
                content = main.get_text(separator='\n', strip=True) if main else ""
                links = []
                for a in soup.find_all('a', href=True):
                    href = self.normalize_url(a['href'])
                    if self.is_valid_url(href):
                        links.append(href)
                if len(content) > 100:
                    return {'url': url, 'title': title, 'content': content, 'links': list(set(links))}
        except:
            pass
        return None

    async def crawl(self):
        to_visit = [self.base_url]
        async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
            while to_visit and len(self.pages) < self.max_pages:
                url = to_visit.pop(0)
                if url in self.visited:
                    continue
                kb.index_progress = f"Crawling {len(self.pages)+1}/{self.max_pages}"
                page = await self.scrape_page(session, url)
                if page:
                    self.pages.append(page)
                    for link in page['links']:
                        if link not in self.visited and link not in to_visit:
                            to_visit.append(link)
                await asyncio.sleep(0.15)
        return self.pages

def chunk_text(text: str, max_len: int = 800) -> List[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) < max_len:
            current += p + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

async def index_documentation(url: str, max_pages: int = 300):
    kb.is_indexing = True
    kb.clear()
    try:
        scraper = DocsScraper(url, max_pages)
        pages = await scraper.crawl()
        kb.pages_count = len(pages)
        total_chunks = sum(len(chunk_text(p['content'])) for p in pages)
        done = 0
        for page in pages:
            chunks = chunk_text(page['content'])
            for i, chunk_content in enumerate(chunks):
                done += 1
                kb.index_progress = f"Embedding {done}/{total_chunks}"
                try:
                    embedding = get_embedding(chunk_content)
                    chunk = WebChunk(url=page['url'], title=page['title'], section=f"Part {i+1}", content=chunk_content, embedding=embedding)
                    kb.add_chunk(chunk)
                except:
                    pass
                await asyncio.sleep(0.05)
        kb.index_progress = f"Done! {len(pages)} pages, {len(kb.chunks)} chunks"
    finally:
        kb.is_indexing = False

SYSTEM_PROMPT = """You are Rork Support, a friendly helpful assistant for Rork - an AI-powered mobile app builder.

Be conversational and helpful. Use simple language. Give clear step-by-step instructions when needed.

If the documentation doesnt cover something, say youre not sure and suggest they contact support.

DOCUMENTATION:
{context}"""

async def answer_question(question: str) -> dict:
    if not kb.chunks:
        return {"answer": "Still loading docs - try again in a minute!", "sources": []}
    try:
        query_emb = get_embedding(question)
        results = kb.search(query_emb, top_k=6)
        if not results:
            return {"answer": "I couldnt find info about that. Try rephrasing or contact support!", "sources": []}
        context = "\n\n---\n\n".join([f"**{c.title}**\n{c.content}" for c in results])
        sources = list({c.url: c for c in results}.values())[:3]
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": question}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return {"answer": response.choices[0].message.content, "sources": [{"url": s.url, "title": s.title} for s in sources]}
    except Exception as e:
        return {"answer": "Something went wrong - try again!", "sources": []}

def format_response(result: dict) -> discord.Embed:
    answer = result.get("answer", "")[:4000]
    embed = discord.Embed(description=answer, color=discord.Color.blue())
    sources = result.get("sources", [])
    if sources:
        txt = "\n".join([f"[{s['title']}]({s['url']})" for s in sources])
        embed.add_field(name="Sources", value=txt[:1000], inline=False)
    return embed

@bot.event
async def on_ready():
    print(f"{bot.user} online!")
    await bot.tree.sync()
    if not kb.chunks and DOCS_URL:
        asyncio.create_task(index_documentation(DOCS_URL, 300))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)
    if bot.user.mentioned_in(message) and not message.mention_everyone:
        q = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if q:
            async with message.channel.typing():
                result = await answer_question(q)
                await message.reply(embed=format_response(result))

@bot.tree.command(name="ask", description="Ask about Rork")
@app_commands.describe(question="Your question")
async def ask_cmd(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    result = await answer_question(question)
    await interaction.followup.send(embed=format_response(result))

@bot.tree.command(name="index", description="Re-index docs")
@app_commands.default_permissions(administrator=True)
async def index_cmd(interaction: discord.Interaction, url: str = None, max_pages: int = 300):
    if kb.is_indexing:
        await interaction.response.send_message(f"Indexing: {kb.index_progress}")
        return
    await interaction.response.send_message(f"Indexing {url or DOCS_URL}...")
    asyncio.create_task(index_documentation(url or DOCS_URL, max_pages))

@bot.tree.command(name="status", description="Bot status")
async def status_cmd(interaction: discord.Interaction):
    e = discord.Embed(title="Status", color=discord.Color.green())
    e.add_field(name="Status", value="Indexing..." if kb.is_indexing else "Ready")
    e.add_field(name="Pages", value=str(kb.pages_count))
    e.add_field(name="Chunks", value=str(len(kb.chunks)))
    await interaction.response.send_message(embed=e)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
