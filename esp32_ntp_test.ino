#include <WiFi.h>
#include <time.h>

// テザリング設定（ここを変更してください）
const char* ssid = "User's iPhone";           // スマホのテザリング名
const char* password = "slotbot2024";         // テザリングパスワード

// NTPサーバー設定
const char* ntpServer = "ntp.nict.jp";        // 日本の公式NTPサーバー
const long gmtOffset_sec = 9 * 3600;          // 日本時間(UTC+9)
const int daylightOffset_sec = 0;             // サマータイム無し

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("===============================");
  Serial.println("ESP32 NTP時刻取得テスト");
  Serial.println("===============================");
  
  // WiFi接続テスト
  testWiFiConnection();
  
  // NTP時刻取得テスト
  if (WiFi.status() == WL_CONNECTED) {
    testNTPSync();
    testTimeDisplay();
  }
  
  // WiFi切断テスト
  testWiFiDisconnect();
  
  // 時刻継続テスト
  testTimeContinuity();
  
  Serial.println("===============================");
  Serial.println("テスト完了");
  Serial.println("===============================");
}

void loop() {
  // 5秒ごとに現在時刻を表示
  delay(5000);
  displayCurrentTime();
}

void testWiFiConnection() {
  Serial.println("--- WiFi接続テスト ---");
  Serial.printf("接続先SSID: %s\n", ssid);
  
  WiFi.begin(ssid, password);
  Serial.print("WiFi接続中");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }
  Serial.println();
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✓ WiFi接続成功！");
    Serial.printf("IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("Signal Strength: %d dBm\n", WiFi.RSSI());
  } else {
    Serial.println("❌ WiFi接続失敗");
    Serial.println("テザリング設定を確認してください:");
    Serial.println("1. スマホのテザリングがONか");
    Serial.println("2. SSID・パスワードが正しいか");
    return;
  }
}

void testNTPSync() {
  Serial.println("--- NTP時刻同期テスト ---");
  Serial.printf("NTPサーバー: %s\n", ntpServer);
  
  // NTP設定
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  
  Serial.print("NTP同期中");
  
  // 時刻取得待ち（最大10秒）
  struct tm timeinfo;
  int attempts = 0;
  while (!getLocalTime(&timeinfo) && attempts < 10) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }
  Serial.println();
  
  if (attempts < 10) {
    Serial.println("✓ NTP時刻同期成功！");
    
    // 取得した時刻を表示
    Serial.println("取得時刻:");
    Serial.printf("  日時: %04d/%02d/%02d %02d:%02d:%02d\n",
                 timeinfo.tm_year + 1900,
                 timeinfo.tm_mon + 1,
                 timeinfo.tm_mday,
                 timeinfo.tm_hour,
                 timeinfo.tm_min,
                 timeinfo.tm_sec);
    
    // Unix timestamp表示
    time_t now = time(nullptr);
    Serial.printf("  Unix timestamp: %ld\n", now);
    Serial.printf("  Unix timestamp(ms): %ld\n", now * 1000);
    
  } else {
    Serial.println("❌ NTP時刻同期失敗");
    Serial.println("インターネット接続を確認してください");
  }
}

void testTimeDisplay() {
  Serial.println("--- 時刻表示テスト ---");
  
  for (int i = 0; i < 5; i++) {
    displayCurrentTime();
    delay(1000);
  }
}

void testWiFiDisconnect() {
  Serial.println("--- WiFi切断テスト ---");
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi切断実行...");
    WiFi.disconnect();
    delay(2000);
    
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("✓ WiFi切断成功");
    } else {
      Serial.println("❌ WiFi切断失敗");
    }
  }
}

void testTimeContinuity() {
  Serial.println("--- 時刻継続性テスト ---");
  Serial.println("WiFi切断後も時刻が継続するかテスト中...");
  
  for (int i = 0; i < 5; i++) {
    Serial.printf("テスト%d回目: ", i + 1);
    displayCurrentTime();
    delay(2000);
  }
  
  Serial.println("✓ 時刻継続性テスト完了");
}

void displayCurrentTime() {
  struct tm timeinfo;
  
  if (getLocalTime(&timeinfo)) {
    // 日本時間で表示
    Serial.printf("現在時刻: %04d/%02d/%02d %02d:%02d:%02d",
                 timeinfo.tm_year + 1900,
                 timeinfo.tm_mon + 1,
                 timeinfo.tm_mday,
                 timeinfo.tm_hour,
                 timeinfo.tm_min,
                 timeinfo.tm_sec);
    
    // Unix timestamp(ms)も表示
    time_t now = time(nullptr);
    Serial.printf(" (Unix: %ld ms)\n", now * 1000);
    
  } else {
    Serial.println("時刻取得失敗 - NTP同期が必要");
  }
}

// ミリ秒精度の時刻取得関数
unsigned long long getCurrentTimeMs() {
  time_t now = time(nullptr);
  return (unsigned long long)now * 1000;
}

// 指定時刻までの待機関数（テスト用）
void waitUntilTime(unsigned long long targetTimeMs) {
  unsigned long long currentTime = getCurrentTimeMs();
  
  if (targetTimeMs > currentTime) {
    unsigned long waitMs = targetTimeMs - currentTime;
    Serial.printf("指定時刻まで%lums待機...\n", waitMs);
    delay(waitMs);
    Serial.println("指定時刻に到達！");
  } else {
    Serial.println("指定時刻は過去の時刻です");
  }
}