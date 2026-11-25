package com.finsistant.v2

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Bundle
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout
import androidx.webkit.WebViewAssetLoader

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var swipeRefresh: SwipeRefreshLayout
    private var activeMobileTab: String = "picks"

    private fun isNetworkAvailable(): Boolean {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return false
        val net = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(net) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        swipeRefresh = findViewById(R.id.swipeRefresh)
        webView = findViewById(R.id.webView)

        swipeRefresh.setColorSchemeColors(
            Color.parseColor("#5B9BFA"),
            Color.parseColor("#21C7A8")
        )
        swipeRefresh.setProgressBackgroundColorSchemeColor(Color.parseColor("#1B222D"))
        swipeRefresh.setOnChildScrollUpCallback { _, _ -> webView.scrollY > 0 }

        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()

        webView.setBackgroundColor(Color.parseColor("#0B0E13"))
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            useWideViewPort = true
            loadWithOverviewMode = true
            builtInZoomControls = false
            displayZoomControls = false
            cacheMode = if (isNetworkAvailable()) WebSettings.LOAD_DEFAULT else WebSettings.LOAD_CACHE_ELSE_NETWORK
            mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
        }

        webView.webChromeClient = WebChromeClient()
        webView.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView,
                request: WebResourceRequest
            ): WebResourceResponse? {
                return assetLoader.shouldInterceptRequest(request.url)
            }

            override fun shouldOverrideUrlLoading(
                view: WebView,
                request: WebResourceRequest
            ): Boolean {
                val url = request.url
                if (url.host == "appassets.androidplatform.net") {
                    if (url.fragment != null) {
                        activeMobileTab = url.fragment ?: "picks"
                    }
                    return false
                }
                startActivity(Intent(Intent.ACTION_VIEW, url))
                return true
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                swipeRefresh.isRefreshing = false
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: WebResourceError
            ) {
                super.onReceivedError(view, request, error)
                if (!request.isForMainFrame) return
                swipeRefresh.isRefreshing = false
                val msg = (error.description ?: "Unable to open terminal").toString()
                    .replace("<", "&lt;").replace(">", "&gt;")
                view.loadDataWithBaseURL(
                    null,
                    """<html><body style="margin:0;background:#0B0E13;color:#E5EAF1;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;padding:24px;text-align:center;">
                       <div><h3 style="margin:0 0 8px 0;color:#5B9BFA;">MOVIQ Terminal Load Error</h3>
                       <p style="margin:0 0 16px 0;color:#8D99AA;font-size:13px;">$msg</p>
                       <button onclick="location.replace('https://appassets.androidplatform.net/assets/index.html')" style="background:#5B9BFA;color:#fff;border:none;border-radius:6px;padding:10px 18px;font-weight:600;">Retry</button></div>
                       </body></html>""".trimIndent(),
                    "text/html",
                    "UTF-8",
                    null
                )
            }

            override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
                swipeRefresh.isRefreshing = false
                recreate()
                return true
            }
        }

        swipeRefresh.setOnRefreshListener {
            webView.settings.cacheMode = if (isNetworkAvailable()) WebSettings.LOAD_DEFAULT else WebSettings.LOAD_CACHE_ELSE_NETWORK
            webView.evaluateJavascript(
                "window.refreshFinsistantMobileScan ? window.refreshFinsistantMobileScan() : location.reload();",
                null
            )
            swipeRefresh.postDelayed({ swipeRefresh.isRefreshing = false }, 1800)
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                webView.evaluateJavascript(
                    "(function(){ try { if (window.handleFinsistantMobileBack && window.handleFinsistantMobileBack()) return 'handled'; if (window.getActiveMobileTab && window.getActiveMobileTab() !== 'mission' && window.switchMobileTab) { window.switchMobileTab('mission'); return 'handled'; } } catch(_){} return 'exit'; })()"
                ) { result ->
                    if (result != "\"handled\"") {
                        isEnabled = false
                        finish()
                    }
                }
            }
        })

        webView.loadUrl("https://appassets.androidplatform.net/assets/index.html")
    }
}
