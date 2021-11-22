from typing import List

DEFAULT_FINANCIALS = [
    "EBITDA",
    "EBIT",
    "EBITDA_MARG",
    "TOTAL_REVENUE",
    "TOTAL_ASSETS",
    "TOTAL_DEBT_EQUITY",
    "FINANCING_CURRENT",
    "LONG_TERM_DEBT",
    "CURRENT_PORTION_DEBT",
    "TOTAL_EQUITY",
    "OPER_INC",
    "NUMBER_EMPLOYEES",
]

HEADERS = {
    "X-API-Key": "ONAI_API_KEY",
    "X-Stack-Host": "Development",
    "X-Request-Id": "-1",
    "X-Stack-User": "Anonymous User",
}


def get_financial_fields_bulk_query(financials: List[str]):
    return (
        """
query($companyIds: [EntityId], $params: TimeSeriesParameterisation) {
    entities: getEntitiesForIds(ids: $companyIds){
         fields(
              mnemonics: ["""
        + " ".join([f'"{el}"' for el in financials])
        + """]
              params: $params
         ) {
            ...latestKeyedValues
         }
    }
}


fragment latestKeyedValues on TimeSeries {
  dataItem {
    mnemonic
  }
  __typename
  ... on BooleanTimeSeries {
    latestDataPoint {
      boolValue: value
    }
  }
  ... on StringTimeSeries {
    latestDataPoint {
      stringValue: value
    }
  }
  ... on IDTimeSeries {
    latestDataPoint {
      idValue: value
    }
  }
  ... on MonetaryAmountTimeSeries {
    dataPoints {
      eventDate
      monetaryAmount {
        currency {
          code
        }
        value: value
      }
    }
  }
  ... on IntegerTimeSeries {
    dataPoints {
      eventDate
      intValue: value
    }
  }
  ... on FloatTimeSeries {
    dataPoints {
      eventDate
      floatValue: value
    }
  }
  ... on StringArrayTimeSeries {
    latestDataPoint {
      stringValues: value
    }
  }
}
"""
    )


GET_FINANCIAL_FIELDS_BULK_QUERY = get_financial_fields_bulk_query(DEFAULT_FINANCIALS)


GET_CONVERSION_RATES_QUERY = """
query($keys: [CurrencyConversionKey]){
  currencyConversionRates(keys: $keys)
  {
    sourceCurrency{
      code
    }
    rate
  }
}
"""
