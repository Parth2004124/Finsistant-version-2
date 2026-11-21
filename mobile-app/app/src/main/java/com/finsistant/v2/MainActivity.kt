package com.finsistant.v2

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.webkit.WebChromeClient
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
            cacheMode = WebSettings.LOAD_DEFAULT
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
        }

        swipeRefresh.setOnRefreshListener {
            webView.evaluateJavascript(
                "window.refreshFinsistantMobileScan ? window.refreshFinsistantMobileScan() : location.reload();",
                null
            )
            swipeRefresh.postDelayed({ swipeRefresh.isRefreshing = false }, 1800)
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                webView.evaluateJavascript(
                    "(function(){ if (window.handleFinsistantMobileBack && window.handleFinsistantMobileBack()) return 'handled'; if (window.getActiveMobileTab && window.getActiveMobileTab() !== 'mission') { window.switchMobileTab('mission'); return 'handled'; } return 'exit'; })()"
                ) { result ->
                    if (result != "\"handled\"") {
                        isEnabled = false
                        onBackPressedDispatcher.onBackPressed()
                    }
                }
            }
        })

        webView.loadUrl("https://appassets.androidplatform.net/assets/index.html")
    }
}
