(() => {
  "use strict";

  const ACTIVE_CLASS = "prx-toc-link--active";

  function decodeFragment(value) {
    try {
      return decodeURIComponent(value.replace(/^#/, ""));
    } catch (_error) {
      return value.replace(/^#/, "");
    }
  }

  function initializeTocSync() {
    if (typeof window.__prxTocCleanup === "function") {
      window.__prxTocCleanup();
    }

    const toc = document.querySelector(
      '.md-sidebar--secondary [data-md-component="toc"]'
    );
    const article = document.querySelector(".md-content__inner");

    if (!toc || !article) {
      window.__prxTocCleanup = undefined;
      return;
    }

    const links = Array.from(toc.querySelectorAll('.md-nav__link[href*="#"]'));
    const linkById = new Map();

    for (const link of links) {
      const fragment = link.hash || link.getAttribute("href")?.split("#")[1];
      if (!fragment) continue;
      linkById.set(decodeFragment(fragment), link);
    }

    const headings = Array.from(article.querySelectorAll("h2[id], h3[id], h4[id], h5[id], h6[id]"))
      .filter((heading) => linkById.has(heading.id));

    if (!headings.length) {
      window.__prxTocCleanup = undefined;
      return;
    }

    const controller = new AbortController();
    const { signal } = controller;
    let frame = 0;
    let hashLockUntil = 0;

    function setActive(id) {
      const activeLink = linkById.get(id);
      if (!activeLink) return false;
      const changed = !activeLink.classList.contains(ACTIVE_CLASS);

      for (const link of links) {
        const active = link === activeLink;
        link.classList.toggle(ACTIVE_CLASS, active);
        if (active) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      }

      if (changed) {
        const scrollArea = activeLink.closest(".md-sidebar__scrollwrap");
        if (scrollArea) {
          const areaBounds = scrollArea.getBoundingClientRect();
          const linkBounds = activeLink.getBoundingClientRect();
          if (linkBounds.top < areaBounds.top) {
            scrollArea.scrollTop -= areaBounds.top - linkBounds.top;
          } else if (linkBounds.bottom > areaBounds.bottom) {
            scrollArea.scrollTop += linkBounds.bottom - areaBounds.bottom;
          }
        }
      }
      return true;
    }

    function contentOffset() {
      const selectors = [".md-header", ".md-tabs"];
      return selectors.reduce((offset, selector) => {
        const element = document.querySelector(selector);
        if (!element || getComputedStyle(element).display === "none") return offset;
        return offset + element.getBoundingClientRect().height;
      }, 24);
    }

    function syncFromScroll() {
      if (performance.now() < hashLockUntil) return;

      const trackedId = decodeFragment(window.location.hash);
      if (trackedId && setActive(trackedId)) return;

      const threshold = contentOffset();
      let current = headings[0];

      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        current = headings[headings.length - 1];
      } else {
        for (const heading of headings) {
          if (heading.getBoundingClientRect().top > threshold) break;
          current = heading;
        }
      }

      setActive(current.id);
    }

    function scheduleScrollSync() {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        syncFromScroll();
      });
    }

    function syncFromHash() {
      const id = decodeFragment(window.location.hash);
      if (id && setActive(id)) {
        hashLockUntil = performance.now() + 500;
        window.setTimeout(scheduleScrollSync, 520);
        return;
      }
      syncFromScroll();
    }

    for (const link of links) {
      link.addEventListener("click", () => {
        const id = decodeFragment(link.hash);
        if (setActive(id)) hashLockUntil = performance.now() + 500;
      }, { signal });
    }

    window.addEventListener("scroll", scheduleScrollSync, { passive: true, signal });
    window.addEventListener("resize", scheduleScrollSync, { passive: true, signal });
    window.addEventListener("hashchange", syncFromHash, { signal });

    syncFromHash();

    window.__prxTocCleanup = () => {
      controller.abort();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }

  if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
    document$.subscribe(initializeTocSync);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeTocSync, { once: true });
  } else {
    initializeTocSync();
  }
})();
