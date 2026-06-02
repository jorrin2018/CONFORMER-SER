#!/usr/bin/env python3
"""
Servidor MCP para integrar Cursor (u otros clientes MCP) con la API REST de GitHub.

Autenticación
-------------
Opciones (en orden de precedencia habitual): variables de entorno del sistema,
o un fichero ``.env`` en la **raíz del repositorio** (mismo nivel que la carpeta
``github_mcp``). Claves reconocidas: ``GITHUB_TOKEN``, ``GH_TOKEN`` o
``github_token``. Usa un Personal Access Token (classic o fine-grained) con los
alcances que necesites (p. ej. ``repo``, ``read:user``).

Ejecución local (stdio, recomendado para Cursor)
------------------------------------------------
Desde la raíz del proyecto, tras instalar dependencias::

    pip install -r github_mcp/requirements.txt
    python -m github_mcp.server

Configuración en Cursor (ejemplo)
---------------------------------
En la configuración MCP del proyecto o del usuario, usa el intérprete Python
donde instalaste ``mcp`` y exporta el token (no lo pongas en el código). El
``cwd`` debe ser el directorio que contiene el paquete ``github_mcp``::

    {
      "mcpServers": {
        "github-personal": {
          "command": "python",
          "args": ["-m", "github_mcp.server"],
          "cwd": "RUTA_AL_PROYECTO",
          "env": {
            "GITHUB_TOKEN": "ghp_..."
          }
        }
      }
    }

Herramientas expuestas: perfil autenticado, listado y detalle de repos,
búsqueda de repositorios, issues, pull requests, contenido de archivos y
creación de issues (operación de escritura).
"""

from __future__ import annotations

import base64
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

GITHUB_API = "https://api.github.com"
DEFAULT_API_VERSION = "2022-11-28"

# Carga .env desde la raíz del repo (no sobreescribe variables ya definidas en el entorno).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

mcp = FastMCP("github_mcp")


class ResponseFormat(str, Enum):
    """Formato de salida para herramientas que devuelven colecciones o vistas."""

    MARKDOWN = "markdown"
    JSON = "json"


def _token() -> str:
    tok = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("github_token")
        or ""
    ).strip()
    if not tok:
        raise RuntimeError(
            "Falta token de GitHub. Define GITHUB_TOKEN, GH_TOKEN o github_token en el entorno "
            "o en un fichero .env en la raíz del proyecto (junto a la carpeta github_mcp)."
        )
    return tok


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": DEFAULT_API_VERSION,
        "User-Agent": "github-mcp-local",
    }


def _handle_api_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        resp = e.response
        try:
            body = resp.json()
            msg = body.get("message", resp.text[:500])
        except Exception:
            msg = resp.text[:500] if resp.text else str(resp.status_code)
        if resp.status_code == 401:
            return (
                "Error: GitHub rechazó el token (401). Comprueba GITHUB_TOKEN/GH_TOKEN, "
                f"caducidad y permisos. Detalle: {msg}"
            )
        if resp.status_code == 403:
            return (
                "Error: Sin permiso o límite de tasa (403). Revisa scopes del token o espera. "
                f"Detalle: {msg}"
            )
        if resp.status_code == 404:
            return f"Error: Recurso no encontrado (404). Detalle: {msg}"
        if resp.status_code == 429:
            return "Error: Límite de peticiones de la API de GitHub. Espera e inténtalo de nuevo."
        return f"Error: GitHub API {resp.status_code}. Detalle: {msg}"
    if isinstance(e, httpx.TimeoutException):
        return "Error: Tiempo de espera agotado al contactar la API de GitHub."
    if isinstance(e, RuntimeError) and "Falta token" in str(e):
        return str(e)
    return f"Error: {type(e).__name__}: {e}"


async def _github_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    url = f"{GITHUB_API}{path}" if path.startswith("/") else f"{GITHUB_API}/{path}"
    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.request(
            method,
            url,
            headers=_headers(),
            params=params,
            json=json_body,
        )
        r.raise_for_status()
        if r.content:
            return r.json()
        return {}


# --- Pydantic inputs ---


class EmptyFormatInput(BaseModel):
    """Opciones mínimas con formato de respuesta."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="markdown para lectura humana; json para procesamiento",
    )


class AuthenticatedUserInput(EmptyFormatInput):
    """Entrada para obtener el usuario autenticado."""


class ListReposInput(BaseModel):
    """Parámetros para listar repositorios del usuario autenticado."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    owner_type: str = Field(
        default="all",
        description="Filtro de propiedad: all, owner, member (parámetro type de la API).",
    )
    sort: str = Field(
        default="updated",
        description="Orden: created, updated, pushed, full_name.",
    )
    direction: str = Field(default="desc", description="asc o desc.")
    per_page: int = Field(default=30, ge=1, le=100, description="Resultados por página (1-100).")
    page: int = Field(default=1, ge=1, description="Número de página.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class GetRepoInput(BaseModel):
    """Owner y nombre del repositorio."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    owner: str = Field(..., min_length=1, description="Usuario u organización (owner).")
    repo: str = Field(..., min_length=1, description="Nombre del repositorio.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ListIssuesInput(BaseModel):
    """Listar issues de un repositorio."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    owner: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    state: str = Field(default="open", description="open, closed, all.")
    per_page: int = Field(default=30, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ListPullRequestsInput(BaseModel):
    """Listat pull requests de un repositorio."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    owner: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    state: str = Field(default="open", description="open, closed, all.")
    per_page: int = Field(default=30, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SearchRepositoriesInput(BaseModel):
    """Búsqueda de repositorios (GitHub search syntax)."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        description="Consulta q para /search/repositories (ej. 'language:python stars:>100').",
    )
    per_page: int = Field(default=20, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class GetFileContentsInput(BaseModel):
    """Obtener contenido de un archivo o listado de un directorio en el repo."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    owner: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    path: str = Field(
        default="",
        description="Ruta dentro del repo; vacío para la raíz.",
    )
    ref: Optional[str] = Field(default=None, description="Rama, tag o SHA opcional.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)
    max_chars: int = Field(
        default=50_000,
        ge=1_000,
        le=500_000,
        description="Máximo de caracteres del fichero decodificado en salida (truncado si excede).",
    )


class CreateIssueInput(BaseModel):
    """Crear un issue en un repositorio."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    owner: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=500)
    body: Optional[str] = Field(default=None, description="Cuerpo Markdown del issue.")
    labels: Optional[List[str]] = Field(default=None, description="Etiquetas por nombre (si existen).")


# --- Tools ---


@mcp.tool(
    name="github_get_authenticated_user",
    annotations={
        "title": "Usuario autenticado en GitHub",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_get_authenticated_user(params: AuthenticatedUserInput) -> str:
    """
    Devuelve el perfil del usuario asociado al token (GET /user).

    Args:
        params: Incluye response_format (markdown|json).

    Returns:
        Texto Markdown o JSON con login, id, nombre público, email si aplica, URL del perfil,
        tipo de cuenta y métricas básicas.
    """
    try:
        data = await _github_request("GET", "/user")
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2, ensure_ascii=False)
        lines = [
            "# Usuario autenticado",
            "",
            f"- **Login**: [{data.get('login')}]({data.get('html_url')}) (id {data.get('id')})",
            f"- **Nombre**: {data.get('name') or '—'}",
            f"- **Bio**: {data.get('bio') or '—'}",
            f"- **Empresa**: {data.get('company') or '—'}",
            f"- **Blog**: {data.get('blog') or '—'}",
            f"- **Repos públicos**: {data.get('public_repos')}",
            f"- **Seguidores / Siguiendo**: {data.get('followers')} / {data.get('following')}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="github_list_repositories",
    annotations={
        "title": "Listar repositorios del usuario",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_list_repositories(params: ListReposInput) -> str:
    """
    Lista repositorios accesibles para el token (GET /user/repos) con paginación.

    Args:
        params: Filtros type/sort/direction, per_page, page y response_format.

    Returns:
        Lista en Markdown o JSON; incluye nombre completo (owner/repo), visibilidad,
        descripción breve, estrellas, lenguaje y fechas relevantes.
    """
    try:
        raw = await _github_request(
            "GET",
            "/user/repos",
            params={
                "type": params.owner_type,
                "sort": params.sort,
                "direction": params.direction,
                "per_page": params.per_page,
                "page": params.page,
            },
        )
        if not isinstance(raw, list):
            return _handle_api_error(RuntimeError("Respuesta inesperada de /user/repos"))
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {
                    "count": len(raw),
                    "page": params.page,
                    "per_page": params.per_page,
                    "items": raw,
                },
                indent=2,
                ensure_ascii=False,
            )
        lines = ["# Repositorios", "", f"Página **{params.page}** — mostrando **{len(raw)}** repos", ""]
        for r in raw:
            full = r.get("full_name", f"{r.get('owner', {}).get('login')}/{r.get('name')}")
            priv = "privado" if r.get("private") else "público"
            lines.append(f"## {full} ({priv})")
            lines.append(f"- **Descripción**: {r.get('description') or '—'}")
            lines.append(f"- **★** {r.get('stargazers_count')} · **Lenguaje**: {r.get('language') or '—'}")
            lines.append(f"- **URL**: {r.get('html_url')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="github_get_repository",
    annotations={
        "title": "Detalle de un repositorio",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_get_repository(params: GetRepoInput) -> str:
    """
    Obtiene metadatos de un repositorio (GET /repos/{owner}/{repo}).

    Args:
        params: owner, repo y response_format.

    Returns:
        Resumen Markdown o JSON completo de la API.
    """
    try:
        data = await _github_request("GET", f"/repos/{params.owner}/{params.repo}")
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2, ensure_ascii=False)
        full = data.get("full_name")
        lines = [
            f"# Repositorio {full}",
            "",
            f"- **Descripción**: {data.get('description') or '—'}",
            f"- **Privado**: {data.get('private')}",
            f"- **Estrellas / Forks**: {data.get('stargazers_count')} / {data.get('forks_count')}",
            f"- **Lenguaje**: {data.get('language') or '—'}",
            f"- **Rama por defecto**: {data.get('default_branch')}",
            f"- **URL**: {data.get('html_url')}",
            f"- **Topics**: {', '.join(data.get('topics') or []) or '—'}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="github_list_issues",
    annotations={
        "title": "Listar issues de un repo",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_list_issues(params: ListIssuesInput) -> str:
    """
    Lista issues del repositorio (GET /repos/{owner}/{repo}/issues). Incluye PRs a menos que filtre la API.

    Args:
        params: owner, repo, state, paginación y response_format.

    Returns:
        Lista numerada en Markdown o JSON con items devueltos por GitHub.
    """
    try:
        raw = await _github_request(
            "GET",
            f"/repos/{params.owner}/{params.repo}/issues",
            params={"state": params.state, "per_page": params.per_page, "page": params.page},
        )
        if not isinstance(raw, list):
            return _handle_api_error(RuntimeError("Respuesta inesperada al listar issues"))
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {"count": len(raw), "page": params.page, "items": raw},
                indent=2,
                ensure_ascii=False,
            )
        lines = [
            f"# Issues en {params.owner}/{params.repo}",
            "",
            f"Estado **{params.state}** — página **{params.page}** ({len(raw)} items)",
            "",
        ]
        for item in raw:
            num = item.get("number")
            title = item.get("title", "")
            state = item.get("state")
            user = (item.get("user") or {}).get("login", "?")
            url = item.get("html_url", "")
            pull = " (pull)" if item.get("pull_request") else ""
            lines.append(f"- **#{num}** [{title}]({url}) — {state} por @{user}{pull}")
        return "\n".join(lines)
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="github_list_pull_requests",
    annotations={
        "title": "Listar pull requests",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_list_pull_requests(params: ListPullRequestsInput) -> str:
    """
    Lista pull requests (GET /repos/{owner}/{repo}/pulls).

    Args:
        params: owner, repo, state, paginación, response_format.

    Returns:
        Tabla Markdown o JSON con número, título, estado, autor y ramas head/base.
    """
    try:
        raw = await _github_request(
            "GET",
            f"/repos/{params.owner}/{params.repo}/pulls",
            params={"state": params.state, "per_page": params.per_page, "page": params.page},
        )
        if not isinstance(raw, list):
            return _handle_api_error(RuntimeError("Respuesta inesperada al listar PRs"))
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {"count": len(raw), "page": params.page, "items": raw},
                indent=2,
                ensure_ascii=False,
            )
        lines = [
            f"# Pull requests en {params.owner}/{params.repo}",
            "",
            f"Estado **{params.state}** — página **{params.page}**",
            "",
        ]
        for pr in raw:
            lines.append(
                f"- **#{pr.get('number')}** [{pr.get('title')}]({pr.get('html_url')}) "
                f"— {pr.get('state')} @{ (pr.get('user') or {}).get('login', '?') } "
                f"(`{pr.get('head', {}).get('ref')}` → `{pr.get('base', {}).get('ref')}`)"
            )
        return "\n".join(lines)
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="github_search_repositories",
    annotations={
        "title": "Buscar repositorios",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_search_repositories(params: SearchRepositoriesInput) -> str:
    """
    Busca repositorios públicos (y accesibles al token) vía GET /search/repositories.

    Args:
        params: query (sintaxis de búsqueda de GitHub), per_page, page, response_format.

    Returns:
        Resumen con total_count y items en Markdown o JSON.
    """
    try:
        data = await _github_request(
            "GET",
            "/search/repositories",
            params={"q": params.query, "per_page": params.per_page, "page": params.page},
        )
        items = data.get("items") or []
        total = data.get("total_count", 0)
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {"total_count": total, "count": len(items), "page": params.page, "items": items},
                indent=2,
                ensure_ascii=False,
            )
        lines = [
            "# Búsqueda de repositorios",
            "",
            f"Consulta: `{params.query}`",
            f"Total aproximado: **{total}** — mostrando **{len(items)}** (página {params.page})",
            "",
        ]
        for r in items:
            lines.append(
                f"- **{r.get('full_name')}** — ★{r.get('stargazers_count')} "
                f"— {r.get('language') or '—'} — [{r.get('html_url')}]({r.get('html_url')})"
            )
        return "\n".join(lines)
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="github_get_file_contents",
    annotations={
        "title": "Contenido de archivo o directorio",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_get_file_contents(params: GetFileContentsInput) -> str:
    """
    Lee un archivo o lista un directorio (GET /repos/{owner}/{repo}/contents/{path}).

    Args:
        params: owner, repo, path, ref opcional, max_chars para truncar texto decodificado.

    Returns:
        Para archivos: contenido texto (truncado) o metadatos; para directorios: listado.
        JSON incluye tipo, sha, tamaño y contenido decodificado cuando aplique.
    """
    try:
        path = params.path.strip().lstrip("/")
        url_path = f"/repos/{params.owner}/{params.repo}/contents/{path}" if path else f"/repos/{params.owner}/{params.repo}/contents"
        req_params: Dict[str, Any] = {}
        if params.ref:
            req_params["ref"] = params.ref
        data = await _github_request("GET", url_path, params=req_params or None)
        if isinstance(data, list):
            if params.response_format == ResponseFormat.JSON:
                return json.dumps({"type": "dir", "items": data}, indent=2, ensure_ascii=False)
            lines = [f"# Contenidos de `{path or '.'}`", ""]
            for item in data:
                t = item.get("type")
                name = item.get("name")
                lines.append(f"- **{name}** ({t})")
            return "\n".join(lines)
        if not isinstance(data, dict):
            return _handle_api_error(RuntimeError("Respuesta inesperada de contents"))
        if data.get("type") != "file":
            return json.dumps(data, indent=2, ensure_ascii=False)
        encoding = data.get("encoding")
        content_b64 = data.get("content")
        text: Optional[str] = None
        if encoding == "base64" and isinstance(content_b64, str):
            raw_bytes = base64.b64decode(content_b64.replace("\n", ""))
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = None
        truncated = False
        if text is not None and len(text) > params.max_chars:
            text = text[: params.max_chars]
            truncated = True
        if params.response_format == ResponseFormat.JSON:
            out = {k: v for k, v in data.items() if k != "content"}
            out["decoded_utf8"] = text
            out["truncated"] = truncated
            return json.dumps(out, indent=2, ensure_ascii=False)
        if text is None:
            return (
                f"# Archivo binario o no UTF-8\n\n- **Ruta**: {data.get('path')}\n"
                f"- **SHA**: {data.get('sha')}\n- **Tamaño**: {data.get('size')} bytes\n"
                f"- Descarga el blob desde la API o el repo si necesitas el binario."
            )
        suffix = "\n\n_(contenido truncado por max_chars)_" if truncated else ""
        return f"# {data.get('path')}\n\n```\n{text}\n```{suffix}"
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="github_create_issue",
    annotations={
        "title": "Crear issue",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def github_create_issue(params: CreateIssueInput) -> str:
    """
    Crea un issue en el repositorio (POST /repos/{owner}/{repo}/issues). Requiere permiso en el repo.

    Args:
        params: owner, repo, title, body opcional, labels opcional.

    Returns:
        JSON con número, URL y estado del issue creado, o mensaje de error accionable.
    """
    try:
        body: Dict[str, Any] = {"title": params.title}
        if params.body is not None:
            body["body"] = params.body
        if params.labels:
            body["labels"] = params.labels
        data = await _github_request(
            "POST",
            f"/repos/{params.owner}/{params.repo}/issues",
            json_body=body,
        )
        return json.dumps(
            {
                "number": data.get("number"),
                "html_url": data.get("html_url"),
                "state": data.get("state"),
                "title": data.get("title"),
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return _handle_api_error(e)


if __name__ == "__main__":
    mcp.run()
