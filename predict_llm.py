import os
import time
import json
from openai import AzureOpenAI, RateLimitError, APIError

AZURE_ENDPOINT    = ""
AZURE_API_KEY     = ""
AZURE_API_VERSION = ""
DEPLOYMENT_ID     = ""

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

def classify_fragment(context: str, punchline: str, max_retries: int = 5) -> str:
    prompt = (
        "Se va a recibir un breve fragmento del transcript de una charla TEDx en América Latina. "
        "La tarea a realizar es determinar si el fragmento fue intencionado para causar humor "
        "(es decir, para provocar la risa del público). La respuesta debe seguir estrictamente este formato:\n"
        "Comienza con ‘HUMOR’ o ‘NO HUMOR’ entre comillas (\"). Luego, la explicación correspondiente entre asteriscos (*). "
        "No debe incluir nada más allá de lo solicitado. La respuesta final no debe exceder las 500 palabras.\n\n"
        f"Contexto: {context}\n"
        f"Punchline: {punchline}\n"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()

        except RateLimitError as e:
            backoff = 2 ** attempt
            print(f"[RateLimit] intento {attempt}/{max_retries}, durmiendo {backoff}s…")
            time.sleep(backoff)

        except APIError as e:
            if getattr(e, "code", "") == "content_filter":
                print("[Filtered] fragmento filtrado por política, marcando NO HUMOR.")
                return '"NO HUMOR" *Filtrado por política*'
            print(f"[APIError] {e.__class__.__name__}: {e}. Marcando NO HUMOR.")
            return '"NO HUMOR" *Error de API*'

        except Exception as e:
            backoff = 2 ** attempt
            print(f"[Error genérico] intento {attempt}/{max_retries}: {e}; durmiendo {backoff}s…")
            time.sleep(backoff)

    print("Reintentos agotados; marcando NO HUMOR.")
    return '"NO HUMOR" *Reintentos agotados*'

def main():
    with open("dataset.jsonl", encoding="utf-8") as fin, \
         open("predictions_llm.jsonl", "a", encoding="utf-8") as fout:
        for line in fin:
            inst = json.loads(line)
            label = classify_fragment(inst["context"], inst["punchline"])
            fout.write(json.dumps({
                "video_id":    inst["video_id"],
                "instance_id": inst["instance_id"],
                "gpt_label":   label
            }, ensure_ascii=False) + "\n")
            fout.flush()
            time.sleep(0.3)

if __name__ == "__main__":
    main()
