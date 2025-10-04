import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Endereço do seu backend Python
    const pythonApiUrl = process.env.PYTHON_API_URL || "http://127.0.0.1:8000/predict";

    // const pythonApiResponse = await fetch(pythonApiUrl, {
    //   method: "POST",
    //   headers: {
    //     "Content-Type": "application/json",
    //   },
    //   body: JSON.stringify(body),
    // });

    console.log(JSON.stringify(body))

    // if (!pythonApiResponse.ok) {
    //   const errorBody = await pythonApiResponse.text();
    //   console.error("Python API Error:", errorBody);
    //   return NextResponse.json(
    //     { error: `Error from analysis model: ${pythonApiResponse.statusText}` },
    //     { status: pythonApiResponse.status }
    //   );
    // }

    // const data = await pythonApiResponse.json();
    
    return NextResponse.json({ message: "Success" });

  } catch (error: any) {
    console.error("API Route Error:", error);
    return NextResponse.json(
      { error: error.message || "An internal server error occurred." },
      { status: 500 }
    );
  }
}
