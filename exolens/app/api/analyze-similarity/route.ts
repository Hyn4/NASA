// app/api/analyze-similarity/route.ts

import { NextResponse } from "next/server";
import { GoogleAuth } from 'google-auth-library';

// Defina a URL da sua API Python aqui. 
// Se estiver rodando localmente, geralmente é 'http://127.0.0.1:8000'.
const ESI_API_URL = process.env.ESI_API_URL || 'http://127.0.0.1:8000/earth_similarity';

// Função para chamar sua API Python de ESI
async function getEsiFromBackend(features: any) {
    const response = await fetch(ESI_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(features),
    });

    console.log(response.body, "aqgor")

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to calculate ESI from Python backend');
    }

    return response.json();
}

async function getExplanationFromGemini(features: any, esiResult: any) {

  const serviceAccountKey = {
      "type": "service_account",
      "project_id": "exolens",
      "private_key_id": "4d2812b0878af5f9a75abf62fd7fe5693e029be6",
      "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDAzK7aGQGIu2QE\ngRb6sstYdgu4tnHD5KvxROZLx5muS2yldmd/CEKkFxLLUEzhlsteoVkML9rXXzJG\n1HjjALjRL7eL56uyehA5yJd4/HSZg3n6ZnmcR6ZCgL90AuAl6pH/Hz9C8L3e3PmW\nLBogTnSqqzOwvrMYa4S1NpoqOWXJqArMGB7xZGhUuPpvXCUWNIWmKy+rmxbK1iyh\nj8FF9iSL5bWSHJoYeHKcdvPRuLCs5Xk4gySyZb2Wrlv0O6vzE5Gkylx/DgLZkYVg\nJSwpz+Z2aAZiE6hqrlmcOnzHmp+Va4DzqQb41E6cKgSVShvhdaTtlTvIxn10HdhK\nJ52zHWtZAgMBAAECggEAFAObeh4C7AIB77rD2yoA1HHKpTXhSPPdyom7u22gvThs\nsp+APm5p1pVjmNIA7SElgvEOaKa2Gcny0un/E5eRV/vTWrVlvD0SHqF9YeyZIQ+G\nM9F7+dZxQrGRTlZ3FNCNL9X7T/RkTXSUfzucSbLDRy1HDCe7uUL+D3630b7MG3sF\nXqyT7IPyNijNCUlf4aOBJLBX5Jxpr+ralNLhhQrQNalgRBOvLT3LfGyFwgKgByVN\n2ASIizQkMLRTA5BwQly3Hc1jz/8UlrV2rgbePb+m8VNBMiWdYqwcLOjA8cvc1IHj\nhoXzbItZNv9PkEJ00NxUr28hJYbW4qwuvHn/8moyQQKBgQDmCTG4dUMEh2hD07zB\nhOGsywA4pLtYwXoPqYnp8l/Q1U30yObfR9tciRuyg8esRLCXh+BiiOubMTVarEk7\nx41gUW9TjBBFp3cYQ14ZMjdUN72dj0yhhQCzE2J6HBpOik3Ht70ByXE3xY7aY+Ui\niiEXLk5DP67+0bxEsNHkRkh7+QKBgQDWj46eDzydh8uLysjb+IaRKlvVJyrDc769\nL88w1u2p/P5Z+cgPRrHZUxv0xmy9fCDJvuuuaEEVURFY9gLx9c9E5sbUGno9tPml\nZX531kBQgfU0s41szgKCg22I+cIWYnRQ9TLxkjWm2b9G64tyZpe/gtVFhMsN9B5M\nmKh20MKCYQKBgGB57q5sD6VwnNwFi56l+ngb04XuINzCmEzFUCAcFO9i5oUJVTrD\nyk5u+nzOJAot7NpAlGz++8Fky/mxVC2MLdD9lnE1xwVPjPVSG775fpcFobLZDMyZ\nGYgBU0XfT3EtNB3VA6IiOCep5ZXWW502zVYJh61QojYhBJLSjdTtXS0RAoGAXDcw\n/Z/g1nfRtNBACcLD20pQU8lUqNJrTRZqPzxwwxmYHAWtxVsF/zioEEjj3YCm+u6S\njtACAO5pvUlmtKWIIr3pAKosla7diQeZFlpAJBnm0HLHOtdD3uIrxq5Ji6NfCJiJ\n/6duZbq6afm8YjvTxpytmwZa2zFrgFIwPXi10KECgYBPosvyuP1CmN5MWUiVpY+r\n30KCcYvxjDtv3F/1tjuJiRMXKVjh1HIjNItxu3IPGjBpbsVDGBTXbBTFU4rJAqqP\nrjDd411WN0SOy50bjqGo4vr78WITME+J4bAuqc8FLU2sVZ13cg/SuOT+VjBGcuSA\n0/+ikp1qVOETbN7lrwRFHA==\n-----END PRIVATE KEY-----\n",
      "client_email": "otavio-augusto@exolens.iam.gserviceaccount.com",
      "client_id": "100989298617681419486",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/otavio-augusto%40exolens.iam.gserviceaccount.com",
      "universe_domain": "googleapis.com"
    }

    const auth = new GoogleAuth({
        credentials: serviceAccountKey,
        scopes: 'https://www.googleapis.com/auth/cloud-platform',
    });
    const accessToken = await auth.getAccessToken();

    // Monta um prompt detalhado para o Gemini
    const prompt = `
    An exoplanet has the following characteristics:
    - Planet Radius: ${features.pl_rade.toFixed(2)} times Earth's radius.
    - Planet Mass: ${features.pl_bmasse.toFixed(2)} times Earth's mass.
    - Surface Temperature: ${features.st_teff}K (Note: This is the star's temperature, the planet's will be lower).
    - Orbital Period: ${features.pl_orbper.toFixed(2)} days.
    - Host Star Radius: ${features.st_rad.toFixed(2)} times the Sun's radius.

    A scientific model calculated its Earth Similarity Index (ESI) as ${(esiResult.earth_similarity_index * 100).toFixed(1)}%.

    Please provide a concise, easy-to-understand explanation for this ESI score. Analyze why the AI likely predicted this value.
    - Start with a direct conclusion about its similarity to Earth.
    - Briefly explain how its key features (size, mass, and implied temperature) compare to Earth's and contribute to the score.
    - Keep the tone scientific but accessible to a general audience.
    - Do not repeat the input values in your explanation. Just analyze them.
    `;

    const GEMINI_API_ENDPOINT = `https://us-central1-aiplatform.googleapis.com/v1/projects/exolens/locations/us-central1/publishers/google/models/gemini-1.0-pro:streamGenerateContent`;

    const response = await fetch(GEMINI_API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            contents: [{
                parts: [{ text: prompt }]
            }],
        }),
    });
    
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Gemini API request failed: ${errorText}`);
    }

    const data = await response.json();
    // Extrai e combina o texto de todas as partes da resposta do Gemini
    const explanation = data.map((chunk: any) => chunk.candidates[0]?.content?.parts[0]?.text || '').join('');

    return explanation;
}


export async function POST(request: Request) {
    try {
        const features = await request.json();

        // 1. Chamar seu backend para obter o score ESI
        const esiResult = await getEsiFromBackend(features);

        // 2. Chamar o Gemini para obter a explicação
        const explanation = await getExplanationFromGemini(features, esiResult);

        // 3. Retornar os dois resultados combinados
        return NextResponse.json({
            success: true,
            esi: esiResult.earth_similarity_index * 100, // Converte para porcentagem
            category: esiResult.similarity_category,
            explanation: explanation,
        });

    } catch (error) {
        console.error("Error in analyze-similarity route:", error);
        return NextResponse.json(
            { success: false, message: error instanceof Error ? error.message : "Unknown error" },
            { status: 500 }
        );
    }
}