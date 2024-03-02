import requests
from datetime import datetime, timezone
from openai import OpenAI

client = OpenAI()
class OSRSScraper:
    def __init__(self):
        self.item_names = self.fetch_item_names()
        self.MIN_PROFIT = 3
        self.MIN_FLUCTUATION = 0
        self.MIN_ROI = 0
        self.MIN_SELL_VOLUME = 100
        self.MIN_BUY_VOLUME = 100

        self.api_url_latest = "https://prices.runescape.wiki/api/v1/osrs/latest"
        self.api_url_5m = "https://prices.runescape.wiki/api/v1/osrs/5m"

    def convert_timestamp(self, ts):
        # Updated to use timezone-aware datetime object
        return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    def fetch_data(self, api_url):
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()['data']
        else:
            print("Failed to retrieve data from the RuneScape API")
            return None

    def fetch_item_names(self):
        mapping_url = "https://prices.runescape.wiki/api/v1/osrs/mapping"
        response = requests.get(mapping_url)
        item_names = {}
        if response.status_code == 200:
            mapping_data = response.json()
            for item in mapping_data:
                item_names[str(item['id'])] = item['name']
        return item_names

    def scrape_data(self):
        data_latest = self.fetch_data(self.api_url_latest)
        data_5m = self.fetch_data(self.api_url_5m)
        items_data = [] 

        if data_latest and data_5m:
            for item_id, item_data_latest in list(data_latest.items())[:40]: 
                if item_id in data_5m:
                    item_data_5m = data_5m[item_id]
                    high_price = item_data_latest.get('high', 0)
                    low_price = item_data_latest.get('low', 0)
                    average_price_5m = item_data_5m.get('avgHighPrice', 0) 
                    average_high_price = item_data_5m.get('avgHighPrice', 0)
                    average_low_price = item_data_5m.get('avgLowPrice', 0)
                    high_price_volume = item_data_5m.get('highPriceVolume', 0)
                    low_price_volume = item_data_5m.get('lowPriceVolume', 0)
                    buy_volume = low_price_volume
                    sell_volume = high_price_volume

                    high_price = average_high_price - average_high_price * 0.01 if average_high_price else 0
                    low_price = int(average_low_price * 0.99) if average_low_price else 0
                    average_price_5m = int(average_price_5m) if average_price_5m else 0

                    if high_price > 0 and low_price > 0: 
                        potential_profit = high_price - low_price
                        profit_margin = (potential_profit / low_price) * 100 

                        if average_price_5m > 0:
                            fluctuation = abs(high_price - average_price_5m) / average_price_5m
                            roi = potential_profit / average_price_5m

                            if (profit_margin >= self.MIN_PROFIT and  
                                fluctuation >= self.MIN_FLUCTUATION and
                                roi >= self.MIN_ROI and
                                sell_volume >= self.MIN_SELL_VOLUME and
                                buy_volume >= self.MIN_BUY_VOLUME):
                                item_name = self.item_names.get(item_id, "Unknown Item")
                                item_data = {
                                    "Item ID": item_id,  # Add this field back in
                                    "Item Name": item_name,
                                    "High (Sell)": high_price,
                                    "High Volume": high_price_volume,
                                    "Low (Buy)": low_price,
                                    "Low Volume": low_price_volume,
                                    "5-Minute Average High Price": average_price_5m,
                                    "ROI": roi,
                                    "Potential Profit": potential_profit,
                                    "Price Fluctuation": fluctuation * 100
                                }
                                items_data.append(item_data) 

                                if len(items_data) >= 40: 
                                    break

        else:
            print("Failed to retrieve data from the RuneScape API")
            return None

        return items_data



class OpenAI_Advisor:

    def response(self, prompt):

        current_gold = "10000"
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a helpful assistant that recieves data from Old School Runescape GE with profit and prices.
                      You current have {current_gold} gold pieces.

                      You only reply with what to buy, how many and for how much each, alongside how much to sell for within your budget.
                        you cant tell them to buy more than they can afford.
                        you cant tell them to buy more than the GE stock.

                     structure your response as follows:
                        1.(first item)"Buy <suggested amount> of <item name> for <suggested buy price> each and sell for <suggested sell price> each"
                        2. (second item)"Buy <suggested amount> of <item name> for <suggested buy price> each and sell for <suggested sell price> each"
                        3. (third item)"Buy <suggested amount> of <item name> for <suggested buy price> each and sell for <suggested sell price> each"
                        4. (fourth item)"Buy <suggested amount> of <item name> for <suggested buy price> each and sell for <suggested sell price> each"
                        5. (fifth item)"Buy <suggested amount> of <item name> for <suggested buy price> each and sell for <suggested sell price> each"
                    
                     
                     """
                },
                {
                    "role": "user",
                    "content": "what should I buy?"+prompt
                }
            ]
        )
        return response.choices[0].message.content
    
    
def format_data_for_prompt(items_data):
    formatted_items = []
    for item in items_data:
        formatted_item = f"Item ID: {item['Item ID']}, Item Name: {item['Item Name']}, High (Sell): {item['High (Sell)']}, Low (Buy): {item['Low (Buy)']}, Potential Profit: {item['Potential Profit']}, ROI: {item['ROI']}, Price Fluctuation: {item['Price Fluctuation']}%"
        formatted_items.append(formatted_item)
    return "\n".join(formatted_items)

# Usage
scraper = OSRSScraper()
items_data = scraper.scrape_data()
formatted_prompt = format_data_for_prompt(items_data)

advisor = OpenAI_Advisor()
response = advisor.response(formatted_prompt)
print(response)
