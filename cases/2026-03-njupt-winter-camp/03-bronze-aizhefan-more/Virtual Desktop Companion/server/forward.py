import asyncio
from aiohttp import web, ClientSession

async def handler(request):
    async with ClientSession() as session:
        async with session.request(
            request.method,
            f"http://localhost:8000{request.path_qs}",
            headers=request.headers,
            data=await request.read()
        ) as resp:
            return web.Response(
                body=await resp.read(),
                status=resp.status,
                headers=resp.headers
            )

app = web.Application()
app.router.add_route("*", "/{path:.*}", handler)
web.run_app(app, host="0.0.0.0", port=9000)
