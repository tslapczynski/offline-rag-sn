# api_connectors.py — Konektory do zewnętrznych API prawniczych
# SAOS: System Analizy Orzeczeń Sądowych (orzeczenia sądowe)
# Sejm ELI: akty prawne z Dziennika Ustaw i Monitora Polskiego

import requests
from langchain_core.documents import Document

# Timeouty — API może być wolne, nie blokujemy appki wiecznie
TIMEOUT = 10  # sekund


class SAOSConnector:
    """
    Publiczne API SAOS — brak klucza/rejestracji.
    Dokumentacja: https://www.saos.org.pl/api/
    """
    BASE_URL = "https://www.saos.org.pl/api/search/judgments"

    def search(self, query: str, n: int = 5) -> list[Document]:
        """
        Szuka orzeczeń pasujących do zapytania.
        Zwraca listę Document zgodną z LangChain.
        """
        params = {
            "all":        query,        # pełnotekstowe
            "courtType":  "SUPREME_COURT",  # Sąd Najwyższy (SN)
            "pageSize":   n,
            "pageNumber": 0,
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            print("  [SAOS] ⚠️ Timeout — serwer nie odpowiedział w czasie")
            return []
        except requests.exceptions.ConnectionError:
            print("  [SAOS] ⚠️ Brak połączenia z internetem lub serwer niedostępny")
            return []
        except Exception as e:
            print(f"  [SAOS] ⚠️ Błąd: {e}")
            return []

        items = data.get("items", [])
        documents = []
        for item in items:
            # Numer sprawy
            case_nums = [c.get("caseNumber", "") for c in item.get("courtCases", [])]
            sygn = ", ".join(filter(None, case_nums)) or "brak sygnatury"

            # Treść (SAOS daje pełny tekst)
            content = item.get("textContent", "").strip()
            if not content:
                continue

            # Skracamy do 2000 znaków per orzeczenie (LLM ma limit kontekstu)
            content_short = content[:2000] + ("..." if len(content) > 2000 else "")

            date = item.get("judgmentDate", "b.d.")
            court_type = item.get("courtType", "")

            text = (
                f"[SAOS — {court_type}] Sygnatura: {sygn} | Data: {date}\n\n"
                f"{content_short}"
            )
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": sygn,
                    "type": "saos_live",
                    "date": date,
                    "url": f"https://www.saos.org.pl/judgments/{item.get('id', '')}",
                }
            ))

        print(f"  [SAOS] Znaleziono {len(documents)} orzeczeń online")
        return documents

    def search_all_courts(self, query: str, n: int = 3) -> list[Document]:
        """Szuka we wszystkich sądach (nie tylko SN)."""
        params = {
            "all":      query,
            "pageSize": n,
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [SAOS all] ⚠️ Błąd: {e}")
            return []

        items = data.get("items", [])
        documents = []
        for item in items:
            case_nums = [c.get("caseNumber", "") for c in item.get("courtCases", [])]
            sygn = ", ".join(filter(None, case_nums)) or "brak sygnatury"
            content = item.get("textContent", "").strip()
            if not content:
                continue
            text = (
                f"[SAOS] Sygnatura: {sygn} | Data: {item.get('judgmentDate', 'b.d.')}\n\n"
                f"{content[:1500]}"
            )
            documents.append(Document(
                page_content=text,
                metadata={"source": sygn, "type": "saos_live"}
            ))

        print(f"  [SAOS all] Znaleziono {len(documents)} orzeczeń")
        return documents


class SejmELIConnector:
    """
    Publiczne API Sejmu — akty prawne z Dz.U. i M.P.
    Dokumentacja: https://api.sejm.gov.pl/eli.html
    Brak klucza — publiczne.
    """
    BASE_URL = "https://api.sejm.gov.pl/eli/acts"

    def search(self, query: str, n: int = 5) -> list[Document]:
        """
        Szuka aktów prawnych pasujących do tytułu/frazy.
        Zwraca listę Document dla LangChain.
        """
        # Szukamy w Dzienniku Ustaw (DU)
        params = {
            "title":     query,
            "publisher": "DU",   # Dziennik Ustaw
            "limit":     n,
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=TIMEOUT)

            # Jeśli endpoint tytułowy nie działa, próbujemy bez filtra
            if resp.status_code == 404:
                resp = requests.get(self.BASE_URL, params={"limit": n}, timeout=TIMEOUT)

            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            print("  [Sejm ELI] ⚠️ Timeout")
            return []
        except requests.exceptions.ConnectionError:
            print("  [Sejm ELI] ⚠️ Brak połączenia")
            return []
        except Exception as e:
            print(f"  [Sejm ELI] ⚠️ Błąd: {e}")
            return []

        # API może zwracać listę lub dict z kluczem "items"
        if isinstance(data, list):
            items = data
        else:
            items = data.get("items", data.get("acts", []))

        documents = []
        for item in items[:n]:
            title     = item.get("title", "Brak tytułu")
            publisher = item.get("publisher", "DU")
            year      = item.get("year", "")
            pos       = item.get("pos", "")
            status    = item.get("status", "")
            eli       = item.get("ELI", "")

            # Budujemy opis aktu (pełna treść jest w PDF — tutaj dajemy metadane)
            text = (
                f"[Sejm ELI — {publisher}] {title}\n"
                f"Rok: {year} | Poz.: {pos} | Status: {status}\n"
                f"Link: https://isap.sejm.gov.pl/eli/{publisher.lower()}/{year}/{pos}/t/isap\n"
                f"ELI: {eli}"
            )

            if title == "Brak tytułu" and not year:
                continue

            documents.append(Document(
                page_content=text,
                metadata={
                    "source": f"{publisher}/{year}/poz.{pos}",
                    "type":   "sejm_eli",
                    "title":  title,
                    "year":   str(year),
                    "url":    f"https://isap.sejm.gov.pl/eli/{publisher.lower()}/{year}/{pos}/t/isap",
                }
            ))

        print(f"  [Sejm ELI] Znaleziono {len(documents)} aktów prawnych")
        return documents

    def get_act_text(self, publisher: str, year: int, pos: int) -> str:
        """Pobiera treść konkretnego aktu (HTML → tekst)."""
        url = f"https://api.sejm.gov.pl/eli/acts/{publisher}/{year}/{pos}/text"
        try:
            resp = requests.get(url, timeout=15)
            if resp.ok:
                # Usuwamy tagi HTML jeśli są
                text = resp.text
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(text, "html.parser")
                    text = soup.get_text(separator="\n")
                except ImportError:
                    pass  # bez bs4 zwracamy raw
                return text[:3000]
        except Exception:
            pass
        return ""
