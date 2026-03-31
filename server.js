import express from 'express';
import multer from 'multer';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleAIFileManager } from '@google/generative-ai/server';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

if (!fs.existsSync('uploads')) fs.mkdirSync('uploads');

const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 200 * 1024 * 1024 }
});

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

const PROMPT = `You are TruthLens — an expert AI fact-checker. Analyze the content provided, use Google Search to find relevant information, then return ONLY a valid JSON object (no markdown, no commentary) with this exact structure:

{
  "verdict": "LEGIT" | "FAKE" | "UNVERIFIED",
  "confidence": <integer 0-100>,
  "content_description": "<what the content shows or says>",
  "summary": "<2-3 sentence summary of findings>",
  "claims": [
    {
      "claim": "<specific claim identified>",
      "verdict": "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIED",
      "explanation": "<evidence-backed explanation>"
    }
  ],
  "fake_info": {
    "applicable": <true if FAKE, else false>,
    "origin": "<where/when this misinformation originated>",
    "likely_started_by": "<who likely created or spread this>",
    "motivation": "<why it was created — political, financial, social, etc.>",
    "how_it_spread": "<how it propagated>",
    "debunked_by": "<credible organizations that have debunked it>"
  },
  "legit_info": {
    "applicable": <true if LEGIT, else false>,
    "full_details": "<comprehensive verified details>",
    "context": "<important historical or social context>",
    "key_facts": ["<fact 1>", "<fact 2>"],
    "supporting_evidence": "<summary of credible supporting evidence>"
  },
  "sources": [
    {
      "title": "<source title>",
      "url": "<source URL>",
      "credibility": "HIGH" | "MEDIUM" | "LOW",
      "snippet": "<brief relevant excerpt>"
    }
  ],
  "search_queries_used": ["<query 1>", "<query 2>"],
  "analysis_notes": "<any caveats or additional context>"
}`;

// Models to try in order (handles per-model quota exhaustion)
const MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.0-flash-lite'];

app.post('/api/analyze', upload.single('file'), async (req, res) => {
  const tempFilePath = req.file?.path;
  try {
    const { type, text } = req.body;

    // Build parts once (shared across model attempts)
    let parts = [];

    if (type === 'text') {
      parts = [{ text: PROMPT + '\n\n---\n\nContent to analyze:\n\n' + text }];
    } else {
      const mimeType = req.file.mimetype;
      console.log(`Uploading ${type}: ${req.file.originalname}`);

      const uploadResult = await fileManager.uploadFile(tempFilePath, {
        mimeType,
        displayName: req.file.originalname,
      });

      if (type === 'video') {
        console.log('Waiting for video processing...');
        let file = await fileManager.getFile(uploadResult.file.name);
        let attempts = 0;
        while (file.state === 'PROCESSING' && attempts < 60) {
          await new Promise(r => setTimeout(r, 5000));
          file = await fileManager.getFile(uploadResult.file.name);
          attempts++;
          console.log(`Video state: ${file.state} (attempt ${attempts})`);
        }
        if (file.state === 'FAILED') {
          throw new Error('Video processing failed. Try a shorter or smaller video.');
        }
      }

      const label = type === 'image' ? 'image' : 'video';
      parts = [
        { text: PROMPT + `\n\n---\n\nAnalyze the following ${label} for misinformation or legitimacy:` },
        { fileData: { fileUri: uploadResult.file.uri, mimeType } }
      ];
    }

    // Try each model until one succeeds
    let response;
    let usedModel;
    for (const modelName of MODELS) {
      try {
        console.log(`Trying model: ${modelName}`);
        const model = genAI.getGenerativeModel({
          model: modelName,
          tools: [{ googleSearch: {} }],
        });
        const result = await model.generateContent(parts);
        response = result.response;
        usedModel = modelName;
        console.log(`Success with model: ${modelName}`);
        break;
      } catch (err) {
        const isQuota = err.message?.includes('429') || err.message?.includes('quota') || err.message?.includes('RESOURCE_EXHAUSTED');
        if (isQuota && MODELS.indexOf(modelName) < MODELS.length - 1) {
          console.warn(`Quota hit on ${modelName}, trying next model...`);
          continue;
        }
        throw err;
      }
    }

    const responseText = response.text();
    console.log(`Analysis complete via ${usedModel}`);

    // Robust JSON extraction
    let parsed;
    try {
      const jsonMatch = responseText.match(/```(?:json)?\n?([\s\S]*?)\n?```/) || [null, responseText.match(/\{[\s\S]*\}/)?.[0]];
      const jsonStr = (jsonMatch[1] || jsonMatch[0] || responseText).trim();
      parsed = JSON.parse(jsonStr);
    } catch (e) {
      console.error('JSON parse failed:', e.message);
      parsed = {
        verdict: 'UNVERIFIED',
        confidence: 50,
        content_description: 'Could not parse structured response',
        summary: responseText.substring(0, 600),
        claims: [],
        fake_info: { applicable: false },
        legit_info: { applicable: false },
        sources: [],
        search_queries_used: [],
        analysis_notes: responseText
      };
    }

    // Merge grounding metadata (search sources from Gemini)
    const groundingMeta = response.candidates?.[0]?.groundingMetadata;
    if (groundingMeta) {
      if (groundingMeta.webSearchQueries?.length) {
        parsed.search_queries_used = [
          ...(parsed.search_queries_used || []),
          ...groundingMeta.webSearchQueries
        ].filter((v, i, a) => a.indexOf(v) === i);
      }
      if (groundingMeta.groundingChunks?.length) {
        const groundedSources = groundingMeta.groundingChunks
          .filter(c => c.web)
          .map(c => ({ title: c.web.title || 'Web Source', url: c.web.uri, credibility: 'MEDIUM', snippet: '' }));
        if (!parsed.sources) parsed.sources = [];
        const existingUrls = new Set(parsed.sources.map(s => s.url));
        groundedSources.forEach(s => { if (!existingUrls.has(s.url)) parsed.sources.push(s); });
      }
    }

    parsed._model_used = usedModel;
    res.json({ success: true, result: parsed });
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ success: false, error: error.message });
  } finally {
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      fs.unlinkSync(tempFilePath);
    }
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n🔍 TruthLens running at http://localhost:${PORT}\n`);
});
