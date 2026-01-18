import os
import re
import asyncio
import discord
from discord import app_commands
from discord.ext import commands
from typing import List, Dict, Tuple, Optional, Set
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
class ScrapedPage:
    url: str
    title: str
    content: str
    headings: List[str]
    links: List[str]

@dataclass
class WebChunk:
    id: str
    url: str
    title: str
    section: str
    content: str
    keywords: List[str]

class WebKnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.sources = {}
        self.is_indexing = False
        self.index_progress = ""
        self.pages_count = 0

    def clear(self):
        self.chunks = []
        self.sources = {}
        self.pages_count = 0

    def _extract_keywords(self, text):
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'has', 'her', 'was', 'one', 'our', 'out', 'had', 'how', 'its', 'may', 'who', 'will', 'with', 'this', 'that', 'have', 'from', 'they', 'been', 'would', 'each', 'which', 'their', 'what', 'there', 'when', 'your', 'just', 'into'}
        return list(set(w for w in words if w not in stop_words))

    def _split_into_sections(self, content):
        sections = []
        current_header = ""
        current_content = []
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                current_header = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))
        return sections

    def add_page(self, page):
        if len(page.content) < 50:
            return
        self.sources[page.url] = page.title
        sections = self._split_into_sections(page.content)
        for i, (section_title, section_content) in enumerate(sections):
            if len(section_content.strip()) < 30:
                continue
            chunk = WebChunk(id=f"{hash(page.url)}_{i}", url=page.url, title=page.title, section=section_title or page.title, content=section_content.strip(), keywords=self._extract_keywords(section_title + " " + section_content))
            self.chunks.append(chunk)

    def search(self, query, top_k=8):
        query_keywords = set(self._extract_keywords(query))
        if not query_keywords:
            return []
        scored_chunks = []
        for chunk in self.chunks:
            chunk_keywords = set(chunk.keywords)
            matches = query_keywords.intersection(chunk_keywords)
            if matches:
                section_keywords = set(self._extract_keywords(chunk.section))
                section_matches = query_keywords.intersection(section_keywords)
                score = len(matches) + (len(section_matches) * 2)
                scored_chunks.append((score, chunk))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

    def get_context_with_sources(self, query, max_tokens=10000):
        chunks = self.search(query, top_k=10)
        if not chunks:
            return "No relevant documentation found.", []
        context_parts = []
        sources = []
        seen_urls = set()
        estimated_tokens = 0
        for chunk in chunks:
            chunk_text = f"### {chunk.section}\n\n{chunk.content}\n\n"
            chunk_tokens = len(chunk_text) // 4
            if estimated_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(chunk_text)
            estimated_tokens += chunk_tokens
            if chunk.url not in seen_urls:
                seen_urls.add(chunk.url)
                sources.append({'url': chunk.url, 'title': chunk.title, 'section': chunk.section})
        return '\n'.join(context_parts), sources

kb = WebKnowledgeBase()

class DocsScraper:
    def __init__(self, base_url, max_pages=200):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visited = set()
        self.pages = []

    def normalize_url(self, url):
        url = url.split('#')[0].rstrip('/')
        if not url.startswith('http'):
            url = urljoin(self.base_url, url)
        return url

    def is_valid_url(self, url):
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False
        skip = ['/assets/', '/static/', '/images/', '/_next/', '.png', '.jpg', '.gif', '.svg', '.css', '.js', '.json']
        for p in skip:
            if p in url.lower():
                return False
        return True

    async def scrape_page(self, session, url):
        url = self.normalize_url(url)
        if url in self.visited:
            return None
        self.visited.add(url)
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status != 200:
                    return None
                if 'text/html' not in response.headers.get('content-type', ''):
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                title = ""
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)
                elif soup.title:
                    title = soup.title.string or ""
                main = soup.find('main') or soup.find('article') or soup.find(class_=re.compile(r'(content|docs|article)', re.I)) or soup.body
                content_parts = []
                headings = []
                if main:
                    for el in main.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'td', 'th']):
                        text = el.get_text(strip=True)
                        if text and len(text) > 5:
                            if el.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                                text = f"\n## {text}\n"
                                headings.append(text)
                            elif el.name == 'li':
                                text = f"* {text}"
                            elif el.name in ['pre', 'code']:
                                text = f"```\n{text}\n```"
                            content_parts.append(text)
                content = '\n'.join(content_parts)
                content = re.sub(r'\n{3,}', '\n\n', content).strip()
                links = []
                for a in soup.find_all('a', href=True):
                    href = self.normalize_url(a['href'])
                    if self.is_valid_url(href):
                        links.append(href)
                if len(content) > 50:
                    return ScrapedPage(url=url, title=title, content=content, headings=headings, links=list(set(links)))
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        return None

    async def crawl(self):
        print(f"Starting crawl of {self.base_url}")
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
                    for link in page.links:
                        if link not in self.visited and link not in to_visit:
                            to_visit.append(link)
                await asyncio.sleep(0.2)
        print(f"Crawl complete: {len(self.pages)} pages")
        return self.pages

async def index_documentation(url, max_pages=200):
    kb.is_indexing = True
    kb.clear()
    try:
        scraper = DocsScraper(url, max_pages)
        pages = await scraper.crawl()
        for page in pages:
            kb.add_page(page)
        kb.pages_count = len(pages)
        kb.index_progress = f"Done! {len(pages)} pages, {len(kb.chunks)} chunks"
    finally:
        kb.is_indexing = False

SYSTEM_PROMPT = """You are Rork Support, a friendly and helpful support agent for Rork - a mobile app development platform that lets anyone build apps using AI.

Your personality:
- Warm, friendly, and conversational
- Use casual language and be encouraging
- Keep responses concise but complete
- If you dont know something, be honest

Guidelines:
- Answer based on the documentation provided below
- Give step-by-step instructions when helpful
- If the docs dont cover something, suggest contacting support
- Sound like a real person, not a robot

DOCUMENTATION:
{context}

Remember: Be helpful, friendly, and human!"""

async def answer_question(question):
    if not kb.chunks:
        return {"answer": "Hey! Im still loading up the docs - give me a minute and try again!", "sources": []}
    context, sources = kb.get_context_with_sources(question, max_tokens=8000)
    if not sources:
        return {"answer": "Hmm, I couldnt find anything specific about that in the docs. Could you try rephrasing or let me know more details?", "sources": []}
    try:
        response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}, {"role": "user", "content": question}], max_tokens=1000, temperature=0.7)
        return {"answer": response.choices[0].message.content, "sources": sources[:3]}
    except Exception as e:
        return {"answer": "Oops, something went wrong! Try again in a sec.", "sources": []}

def format_response(result):
    answer = result.get("answer", "No answer found.")
    if len(answer) > 4000:
        answer = answer[:3997] + "..."
    embed = discord.Embed(description=answer, color=discord.Color.blue())
    sources = result.get("sources", [])
    if sources:
        source_text = "\n".join([f"[{s['title']}]({s['url']})" for s in sources])
        embed.add_field(name="Learn more:", value=source_text[:1000], inline=False)
    return embed

@bot.event
async def on_ready():
    print(f"{bot.user} is online!")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} commands")
    except Exception as e:
        print(f"Failed to sync: {e}")
    if not kb.chunks and DOCS_URL:
        asyncio.create_task(index_documentation(DOCS_URL, max_pages=200))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)
    if bot.user.mentioned_in(message) and not message.mention_everyone:
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if question:
            async with message.channel.typing():
                result = await answer_question(question)
                embed = format_response(result)
                await message.reply(embed=embed)

@bot.tree.command(name="ask", description="Ask about Rork")
@app_commands.describe(question="Your question")
async def ask_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    result = await answer_question(question)
    await interaction.followup.send(embed=format_response(result))

@bot.tree.command(name="index", description="Re-index docs")
@app_commands.default_permissions(administrator=True)
async def index_command(interaction: discord.Interaction, url: str = None, max_pages: int = 200):
    if kb.is_indexing:
        await interaction.response.send_message(f"Already indexing: {kb.index_progress}")
        return
    await interaction.response.send_message(f"Indexing {url or DOCS_URL}...")
    asyncio.create_task(index_documentation(url or DOCS_URL, max_pages))

@bot.tree.command(name="status", description="Bot status")
async def status_command(interaction: discord.Interaction):
    embed = discord.Embed(title="Status", color=discord.Color.green())
    embed.add_field(name="Status", value="Indexing..." if kb.is_indexing else "Ready", inline=False)
    embed.add_field(name="Pages", value=str(kb.pages_count), inline=True)
    embed.add_field(name="Chunks", value=str(len(kb.chunks)), inline=True)
    await interaction.response.send_message(embed=embed)

if __name__ == "__main__":
    if not DISCORD_TOKEN or not OPENAI_API_KEY:
        print("Missing DISCORD_TOKEN or OPENAI_API_KEY")
        exit(1)
    bot.run(DISCORD_TOKEN)

