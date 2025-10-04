import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Endereço do seu backend Python
    const pythonApiUrl = process.env.PYTHON_API_URL || "http://127.0.0.1:8000/predict";

    const pythonApiResponse = await fetch(pythonApiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    let responseText = "";
    if (pythonApiResponse.body) {
      const reader = pythonApiResponse.body.getReader();
      const decoder = new TextDecoder("utf-8");
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        responseText += decoder.decode(value, { stream: true });
      }
      responseText += decoder.decode(); // flush
      // Se quiser, pode fazer JSON.parse(responseText) se for JSON
      // const data = JSON.parse(responseText);
      // return NextResponse.json(data);
      console.log("Full response from Python API:", responseText);
      return NextResponse.json({ raw: responseText });
    }else {
  return NextResponse.json({ error: "No response body" }, { status: 500 });
}

  } catch (error: any) {
    console.error("API Route Error:", error);
    return NextResponse.json(
      { error: error.message || "An internal server error occurred." },
      { status: 500 }
    );
  }
}
