(() => {
    "use strict";

    const root = document.documentElement;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const isReducedMotion = () => reducedMotion.matches;
    const supportsIntersectionObserver = "IntersectionObserver" in window;

    if (!isReducedMotion()) {
        root.classList.add("js-motion");
    }

    const revealItems = document.querySelectorAll(".reveal");
    if (supportsIntersectionObserver && !isReducedMotion()) {
        const revealObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.12 });
        revealItems.forEach((item) => revealObserver.observe(item));
    } else {
        revealItems.forEach((item) => item.classList.add("is-visible"));
    }

    const navToggle = document.querySelector(".nav-toggle");
    const navLinks = document.querySelector(".nav-links");
    if (navToggle && navLinks) {
        const closeNavigation = () => {
            navToggle.setAttribute("aria-expanded", "false");
            navToggle.setAttribute("aria-label", "Open navigation");
            navLinks.classList.remove("is-open");
        };
        navToggle.addEventListener("click", () => {
            const isOpen = navToggle.getAttribute("aria-expanded") === "true";
            navToggle.setAttribute("aria-expanded", String(!isOpen));
            navToggle.setAttribute("aria-label", isOpen ? "Open navigation" : "Close navigation");
            navLinks.classList.toggle("is-open", !isOpen);
        });
        navLinks.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeNavigation));
        document.addEventListener("click", (event) => {
            if (!navLinks.contains(event.target) && !navToggle.contains(event.target)) {
                closeNavigation();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") closeNavigation();
        });
    }

    const hero = document.querySelector(".hero");
    const backgroundVideo = document.querySelector(".hero-background-video");
    const videoToggle = document.querySelector("[data-video-toggle]");
    const videoToggleLabel = document.querySelector("[data-video-toggle-label]");
    const wideScreen = window.matchMedia("(min-width: 768px)");
    let heroIsVisible = true;
    let backgroundVideoPaused = false;
    let backgroundVideoFailed = false;
    let videoIsActive = false;
    let resumeNetwork = () => {};
    let pauseNetwork = () => {};
    try {
        backgroundVideoPaused = window.localStorage.getItem("fraudguard-background-video-paused") === "true";
    } catch {
        backgroundVideoPaused = false;
    }

    if (hero && backgroundVideo && videoToggle) {
        const updateVideoToggle = () => {
            const canPlayVideo = wideScreen.matches && !isReducedMotion() && !backgroundVideoFailed;
            videoToggle.hidden = !canPlayVideo;
            const isPlaying = !backgroundVideoPaused && !backgroundVideo.paused;
            videoToggle.setAttribute("aria-pressed", String(isPlaying));
            if (videoToggleLabel) {
                videoToggleLabel.textContent = isPlaying ? "Pause background" : "Play background";
            }
        };
        const setVideoActive = (active) => {
            videoIsActive = active;
            hero.classList.toggle("is-video-active", active);
            if (active) pauseNetwork();
            else resumeNetwork();
            updateVideoToggle();
        };
        const syncBackgroundVideo = () => {
            const canLoadVideo = wideScreen.matches && !isReducedMotion() && !backgroundVideoFailed;
            if (!canLoadVideo) {
                backgroundVideo.pause();
                backgroundVideo.removeAttribute("src");
                backgroundVideo.load();
                setVideoActive(false);
                updateVideoToggle();
                return;
            }
            const shouldPlay = wideScreen.matches && !isReducedMotion() && !backgroundVideoPaused &&
                heroIsVisible && !document.hidden && !backgroundVideoFailed;
            updateVideoToggle();
            if (!shouldPlay) {
                backgroundVideo.pause();
                setVideoActive(false);
                return;
            }
            if (!backgroundVideo.getAttribute("src")) {
                backgroundVideo.setAttribute("src", backgroundVideo.dataset.src);
            }
            const playAttempt = backgroundVideo.play();
            if (playAttempt && typeof playAttempt.then === "function") {
                playAttempt.then(() => {
                    if (!backgroundVideo.paused) setVideoActive(true);
                }).catch(() => setVideoActive(false));
            }
        };

        backgroundVideo.addEventListener("error", () => {
            backgroundVideoFailed = true;
            backgroundVideo.pause();
            setVideoActive(false);
            updateVideoToggle();
        });
        videoToggle.addEventListener("click", () => {
            backgroundVideoPaused = !backgroundVideoPaused;
            try {
                window.localStorage.setItem("fraudguard-background-video-paused", String(backgroundVideoPaused));
            } catch {
            }
            syncBackgroundVideo();
        });
        if (supportsIntersectionObserver) {
            const videoObserver = new IntersectionObserver((entries) => {
                heroIsVisible = entries[0].isIntersecting;
                syncBackgroundVideo();
            }, { threshold: 0.01 });
            videoObserver.observe(hero);
        }
        wideScreen.addEventListener?.("change", syncBackgroundVideo);
        reducedMotion.addEventListener?.("change", syncBackgroundVideo);
        document.addEventListener("visibilitychange", syncBackgroundVideo);
        syncBackgroundVideo();
    }

    const networkSvg = document.querySelector(".network-svg");
    if (networkSvg) {
        const edges = Array.from(networkSvg.querySelectorAll(".network-edge"));
        const nodes = Array.from(networkSvg.querySelectorAll(".network-node"));
        const edgeSequence = ["0-1", "1-2", "2-3", "3-6", "6-4", "4-7", "7-0"];
        const nodeSequence = ["0", "1", "2", "3", "6", "4", "7"];
        let networkTimer = 0;
        const clearNetwork = () => {
            window.clearTimeout(networkTimer);
            edges.forEach((edge) => edge.classList.remove("is-hot"));
            nodes.forEach((node) => node.classList.remove("is-hot", "is-active"));
        };
        const showStaticNetwork = () => {
            clearNetwork();
            networkSvg.querySelector('[data-edge="0-1"]')?.classList.add("is-hot");
            networkSvg.querySelector('[data-edge="1-2"]')?.classList.add("is-hot");
            networkSvg.querySelector('[data-node="2"]')?.classList.add("is-hot");
        };
        const runNetwork = () => {
            if (videoIsActive) return;
            clearNetwork();
            if (isReducedMotion()) {
                showStaticNetwork();
                return;
            }
            if (document.hidden) return;
            let step = 0;
            const pulse = () => {
                if (document.hidden || isReducedMotion()) return;
                if (step >= edgeSequence.length) {
                    networkTimer = window.setTimeout(runNetwork, 1250);
                    return;
                }
                const edge = networkSvg.querySelector(`[data-edge="${edgeSequence[step]}"]`);
                const node = networkSvg.querySelector(`[data-node="${nodeSequence[step]}"]`);
                if (edge) edge.classList.add("is-hot");
                if (node) node.classList.add(step === 0 ? "is-hot" : "is-active");
                step += 1;
                networkTimer = window.setTimeout(pulse, 610);
            };
            networkTimer = window.setTimeout(pulse, 500);
        };
        resumeNetwork = runNetwork;
        pauseNetwork = clearNetwork;
        runNetwork();
        document.addEventListener("visibilitychange", () => {
            if (document.hidden) clearNetwork();
            else runNetwork();
        });
        reducedMotion.addEventListener?.("change", runNetwork);
    }

    const featureTabs = Array.from(document.querySelectorAll(".feature-tab"));
    let activeTabIndex = 0;
    let tabTimer = 0;
    const showFeatureTab = (index) => {
        if (!featureTabs.length) return;
        activeTabIndex = (index + featureTabs.length) % featureTabs.length;
        featureTabs.forEach((tab, tabIndex) => {
            const active = tabIndex === activeTabIndex;
            tab.classList.toggle("is-active", active);
            tab.setAttribute("aria-selected", String(active));
            tab.tabIndex = active ? 0 : -1;
            const panel = document.getElementById(tab.getAttribute("aria-controls"));
            if (panel) {
                panel.hidden = !active;
                panel.classList.toggle("is-active", active);
            }
        });
    };
    const stopTabTimer = () => window.clearInterval(tabTimer);
    const startTabTimer = () => {
        stopTabTimer();
        if (featureTabs.length > 1 && !isReducedMotion() && !document.hidden) {
            tabTimer = window.setInterval(() => showFeatureTab(activeTabIndex + 1), 5500);
        }
    };
    featureTabs.forEach((tab, index) => {
        tab.addEventListener("click", () => {
            showFeatureTab(index);
            startTabTimer();
        });
        tab.addEventListener("keydown", (event) => {
            let nextIndex = index;
            if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % featureTabs.length;
            else if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + featureTabs.length) % featureTabs.length;
            else if (event.key === "Home") nextIndex = 0;
            else if (event.key === "End") nextIndex = featureTabs.length - 1;
            else return;
            event.preventDefault();
            showFeatureTab(nextIndex);
            featureTabs[nextIndex].focus();
            startTabTimer();
        });
    });
    const tabsContainer = document.querySelector(".feature-tabs");
    tabsContainer?.addEventListener("pointerenter", stopTabTimer);
    tabsContainer?.addEventListener("pointerleave", startTabTimer);
    tabsContainer?.addEventListener("focusin", stopTabTimer);
    tabsContainer?.addEventListener("focusout", (event) => {
        if (!tabsContainer.contains(event.relatedTarget)) startTabTimer();
    });
    showFeatureTab(0);
    startTabTimer();

    const explainer = document.querySelector(".explainer-dialog");
    if (explainer) {
        const closeButton = explainer.querySelector(".dialog-close");
        const nextButton = explainer.querySelector(".explainer-next");
        const slides = Array.from(explainer.querySelectorAll(".explainer-slide"));
        const slideCount = explainer.querySelector(".slide-count");
        let activeSlide = 0;
        let slideTimer = 0;
        const showSlide = (index) => {
            activeSlide = (index + slides.length) % slides.length;
            slides.forEach((slide, slideIndex) => {
                const active = slideIndex === activeSlide;
                slide.classList.toggle("is-current", active);
                slide.hidden = !active;
            });
            slideCount.textContent = `${String(activeSlide + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
            nextButton.innerHTML = activeSlide === slides.length - 1 ? 'Done <span aria-hidden="true">✓</span>' : 'Next <span aria-hidden="true">→</span>';
            explainer.classList.remove("is-playing");
            void explainer.offsetWidth;
            explainer.classList.add("is-playing");
        };
        const stopSlides = () => {
            window.clearInterval(slideTimer);
            explainer.classList.remove("is-playing");
        };
        const startSlides = () => {
            stopSlides();
            if (explainer.open && !isReducedMotion() && !document.hidden) {
                explainer.classList.add("is-playing");
                slideTimer = window.setInterval(() => showSlide(activeSlide + 1), 5000);
            }
        };
        const openExplainer = () => {
            showSlide(0);
            explainer.showModal();
            startSlides();
        };
        document.querySelectorAll("[data-open-explainer]").forEach((button) => button.addEventListener("click", openExplainer));
        closeButton.addEventListener("click", () => explainer.close());
        nextButton.addEventListener("click", () => {
            if (activeSlide === slides.length - 1) explainer.close();
            else {
                showSlide(activeSlide + 1);
                startSlides();
            }
        });
        explainer.addEventListener("close", stopSlides);
        explainer.addEventListener("keydown", (event) => {
            if (event.key !== "Tab") return;
            const focusable = Array.from(explainer.querySelectorAll("button:not([disabled]), a[href], [tabindex]:not([tabindex='-1'])"))
                .filter((element) => !element.hidden);
            if (!focusable.length) return;
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        });
        document.addEventListener("visibilitychange", () => {
            if (document.hidden) stopSlides();
            else startSlides();
        });
        reducedMotion.addEventListener?.("change", startSlides);
    }

    const demoResult = document.querySelector(".demo-result");
    const demoButtons = Array.from(document.querySelectorAll(".scenario-button"));
    const demoEmpty = document.querySelector("[data-demo-empty]");
    const demoOutput = document.querySelector("[data-demo-output]");
    const demoStatus = document.querySelector("[data-demo-status]");
    if (demoResult && demoButtons.length && demoEmpty && demoOutput && demoStatus) {
        const decision = demoOutput.querySelector("[data-decision]");
        const score = demoOutput.querySelector("[data-risk-score]");
        const reasons = demoOutput.querySelector("[data-reasons]");
        const presentResult = (result) => {
            demoEmpty.hidden = true;
            demoOutput.hidden = false;
            decision.textContent = result.decision;
            decision.className = "decision-badge";
            if (/hold/i.test(result.decision)) decision.classList.add("is-hold");
            else if (/review/i.test(result.decision)) decision.classList.add("is-review");
            const numericScore = Number(result.risk_score);
            score.textContent = Number.isFinite(numericScore) ? numericScore.toFixed(2) : "Unavailable";
            reasons.replaceChildren();
            result.reasons.forEach((reason) => {
                const item = document.createElement("li");
                item.textContent = reason;
                reasons.append(item);
            });
        };
        demoButtons.forEach((button) => {
            button.addEventListener("click", async () => {
                demoButtons.forEach((candidate) => {
                    candidate.disabled = true;
                    candidate.setAttribute("aria-pressed", String(candidate === button));
                });
                demoResult.setAttribute("aria-busy", "true");
                demoStatus.textContent = "Scoring the simulated profile…";
                try {
                    const response = await fetch(demoResult.dataset.endpoint, {
                        method: "POST",
                        headers: { "Content-Type": "application/json", "Accept": "application/json" },
                        credentials: "same-origin",
                        body: JSON.stringify({ scenario: button.dataset.scenario })
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        if (response.status === 429) throw new Error("Demo limit reached. Please try again in a minute.");
                        throw new Error(data.error || "The demo could not complete this assessment.");
                    }
                    if (typeof data.decision !== "string" || !Array.isArray(data.reasons)) {
                        throw new Error("The demo returned an unexpected response.");
                    }
                    presentResult(data);
                    demoStatus.textContent = "Prediction complete. This simulated result was not saved.";
                } catch (error) {
                    demoStatus.textContent = error instanceof Error ? error.message : "The demo is temporarily unavailable.";
                } finally {
                    demoButtons.forEach((candidate) => { candidate.disabled = false; });
                    demoResult.removeAttribute("aria-busy");
                }
            });
        });
    }

    const metricItems = Array.from(document.querySelectorAll(".metric-number"));
    const formatMetric = (value, decimals, suffix) => `${value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;
    const animateMetric = (element) => {
        if (element.dataset.animated === "true") return;
        const target = Number(element.dataset.value);
        if (!Number.isFinite(target)) return;
        element.dataset.animated = "true";
        const suffix = element.dataset.suffix || "";
        const decimals = (element.dataset.value.split(".")[1] || "").length;
        if (isReducedMotion()) {
            element.textContent = formatMetric(target, decimals, suffix);
            return;
        }
        const start = performance.now();
        const duration = 1150;
        const step = (now) => {
            const progress = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - progress, 3);
            element.textContent = formatMetric(target * eased, decimals, suffix);
            if (progress < 1) window.requestAnimationFrame(step);
            else element.textContent = formatMetric(target, decimals, suffix);
        };
        window.requestAnimationFrame(step);
    };
    if (supportsIntersectionObserver && !isReducedMotion()) {
        const metricObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    animateMetric(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });
        metricItems.forEach((item) => metricObserver.observe(item));
    } else {
        metricItems.forEach(animateMetric);
    }

    const techTrack = document.querySelector(".tech-track");
    const setHiddenTabMotion = () => {
        if (techTrack) techTrack.style.animationPlayState = document.hidden ? "paused" : "";
        if (document.hidden) stopTabTimer();
        else startTabTimer();
        if (!document.hidden) resumeNetwork();
    };
    document.addEventListener("visibilitychange", setHiddenTabMotion);
})();
