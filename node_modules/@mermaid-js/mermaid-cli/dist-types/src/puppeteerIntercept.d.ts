/**
 * Puppeteer doesn't allow importing ESM modules from `file://` URLs.
 * We don't want to create a dummy http server to serve ESM modules
 * (since that would cause issues with ports/firewalls), so this module
 * instead intercepts dummy `https://mermaid-cli-intercept.invalid` requests.
 */
export class Interceptor {
    /**
       * @param {URL | `file://${string}`} fileUrl - File URL
       */
    fileUrlToInterceptUrl(fileUrl: URL | `file://${string}`): Promise<string>;
    /**
       *
       * @param {URL | string} interceptUrl
       * @throws {Error} If the URL is not a valid intercept URL
       */
    interceptUrlToFileUrl(interceptUrl: URL | string): Promise<URL>;
    /**
       * Intercepts requests to `https://mermaid-cli-intercept.invalid`
       * and serves the corresponding file content.
       *
       * @return {puppeteer.Handler<puppeteer.HTTPRequest>}
       */
    get interceptRequestHandler(): puppeteer.Handler<puppeteer.HTTPRequest>;
    #private;
}
import type puppeteer from 'puppeteer';
//# sourceMappingURL=puppeteerIntercept.d.ts.map