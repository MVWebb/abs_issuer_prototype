# Run 'pip install -r requirements' in order to get the pyba module and dependencies
from pyba import run
import asyncio
import sfportal
import pandas as pd
import numpy as np
import decimal
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize


async def main(pyba):
    # pyba.init_excel()

    # await getKeyStats("KeyStats","Portfolio - KeyStats","A2","B4","B9","A10","B10")

    await getPoolHist("PoolHistory", "Portfolio - PoolHist", "A2", "A5", "A7",
                      await pyba.read_range("PoolHistory", "B4"),
                      await pyba.read_range("PoolHistory", "E4"), await pyba.read_range("PoolHistory", "H4"),
                      await pyba.read_range("PoolHistory", "K4"), pyba)


async def getPoolHist(wsName, portfolio, portfolioStart, headersStart, resStart, lookBack, displayGraphs, sameIssuer,
                      compare, pyba):
    '''

    :param wsName: The Worksheet name the code should read and write to.
    :param headersStart: The cell location where the headers start.
    :param inputStart: Where the required IDs used to retrieve data starts.
    :param resStart: The location you would like the results to be populated to.
    :param lookBack: By default this is set to 1, this will retrieve a specified number of months of data
    :return:
    '''
    # Look into adding vintage as a criteria
    i = 0
    x = 0
    counter = 0
    df = pd.DataFrame()
    lst_cDates = {}

    sfp = sfportal.SFPortal()
    await sfp.login('webbm', 'sav2015')
    await pyba.log("Gathering Required Fields..")

    fields = [field for field in await pyba.read_range(wsName, headersStart, columns=50) if len(field) > 0]
    fields.append('closing_date')
    await pyba.log("Fields Gathered...")
    await pyba.log("Gathering Requested Deals..")
    df1 = pd.DataFrame(columns=fields)

    cusips = [cusip for cusip in await pyba.read_range_to_bottom(portfolio, portfolioStart)]
    await pyba.log("Deals Gathered...")

    srs = await sfportal.SearchCusipIsin.for_cusipisins(sfp, cusips)

    async def hist_query(sr):
        hist = await sfportal.History.for_deal(sfp, "ABS", sr.matched_result["deal_key"])
        return sr, hist

    sr_hists = await asyncio.gather(*[hist_query(sr) for sr in srs])

    async def cDate_query(sr):
        ks = await sfportal.KeyStats.for_search_results(sfp, [sr])
        return sr, ks

    sr_cDates = await asyncio.gather(*[cDate_query(sr) for sr in srs])

    for sr, ks in sr_cDates:
        try:
            df['Closing Date'] = ks.results[0]['$']['closing_date']
        except (IndexError):
            continue
        lst_cDates[ks.results[0]['$']['deal_key']] = ks.results[0]['$']['closing_date']

    for sr, hist in sr_hists:
        try:
            hist.results[0]['Series']
        except (KeyError, IndexError):
            continue
        if lookBack == 'All' or counter == 1:
            cnt = sum([1 for key in hist.results[0]['Series'] if '$' in key])
            lookBack = cnt
            counter = 1
        else:
            lookBack = int(lookBack)

        # Converting json file to a Dataframe
        df = json_normalize(hist.results[0]['Series'][:lookBack])
        # Removing columns not specified in the Excel
        df.columns = df.columns.str.lstrip('$.')
        colDrop = [cols for cols in df.columns if cols not in fields]
        df = df.drop(colDrop, axis=1)
        # Adding necessary columns
        cusip = sr.matched_result["cusip1"]
        df['Deal ID'], df['dealname'], df['cusip'] = [sr.matched_result["deal_key"], sr.matched_result["dealname"],
                                                      cusip]

        df1 = df1.append(df, sort=True)

    df1.drop_duplicates(inplace=True)

    # Adding Closing Date to the DataFrame and calculating Seasoning, Seasoning is difficult due to how Python 3 handles
    # rounding.

    df1['closing_date'] = df1['Deal ID'].map(lst_cDates)
    df1 = df1.reindex(columns=fields).sort_values(by=['dealname', 'date'], kind='mergesort')
    df1['vintage'] = pd.to_datetime(df1['closing_date']).dt.year
    df1['closing_date'], df1['date'] = [pd.to_datetime(df1['closing_date']), pd.to_datetime(df1['date'])]

    seas_calc = (df1['date'] - df1['closing_date']) / np.timedelta64(1, 'M')
    seas_calc1 = [int(decimal.Decimal(seas).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_EVEN)) for seas
                  in seas_calc]
    df1['seasoning_months'] = seas_calc1[:]
    df1['seasoning1'] = (df1['date'] - df1['closing_date']) / np.timedelta64(1, 'M')

    cols = ['CUMCHARGEOFF', 'NETCHARGEOFFSAMT', 'GROSSCHARGEOFFSAMT', 'CPR1MTH', 'CDR1MTH', 'POOLFACTOR',
            'ORA_60PDAYRATE', 'CUMGROSSCHARGEOFF', 'CUMMCHARGEOFFSAMT', 'orig_balance']
    df1[cols] = df1[cols].apply(pd.to_numeric)

    pd_legend = {'loc': 'upper center', 'fancybox': True, 'shadow': True}
    pd_xlabel = {'xlabel': 'Deal Age (Months)', 'fontsize': '15'}
    pd_plot = {'linewidth': '4', 'fontsize': '15', 'grid': True}

    if displayGraphs == 'Yes' and sameIssuer == 'Yes' and compare == 'No':
        # Creating a Cumulative Net Loss Graph that represents each deal
        await pyba.log("Creating CNL % Graph...")
        cnl_df = df1.pivot_table(index='seasoning_months', columns='dealname', values='CUMCHARGEOFF')
        cnl_df['Average'] = cnl_df.mean(axis=1)
        fig, ax = plt.subplots()

        fig.suptitle('Cumulative Net Loss % By Deal', fontsize=20)
        ax = cnl_df.plot(colormap='Set1', ax=ax, **pd_plot)
        ax.lines[-1].set_linestyle("--")
        ax.lines[-1].set_color("black")
        plt.xlabel(**pd_xlabel)
        plt.ylabel('CNL %', fontsize=15)
        plt.legend(prop={'size': 8}, ncol=3, **pd_legend)
        plt.show()

        # Creating a Loss Severity Graph that represents each vintage
        await pyba.log("Creating Loss Severity % Graph...")
        lossSev_df = df1[['seasoning_months', 'vintage', 'NETCHARGEOFFSAMT', 'GROSSCHARGEOFFSAMT']]
        lossSev_df = lossSev_df.reset_index(drop=True)
        lossSev_df = (lossSev_df.groupby(['seasoning_months', 'vintage'])['NETCHARGEOFFSAMT'].sum() /
                      lossSev_df.groupby(['seasoning_months', 'vintage'])['GROSSCHARGEOFFSAMT'].sum()) * 100
        lossSev_df = lossSev_df.unstack()
        lossSev_df['Average'] = lossSev_df.mean(axis=1)
        fig, ax = plt.subplots()

        fig.suptitle('Loss Severity By Vintage', fontsize=20)
        ax = lossSev_df.plot(ax=ax, colormap='Set1', **pd_plot)
        ax.lines[-1].set_linestyle("--")
        ax.lines[-1].set_color("black")
        plt.ylabel('Loss Severity %', fontsize=15)
        plt.xlabel(**pd_xlabel)
        plt.legend(ncol=5, prop={'size': 15}, **pd_legend)
        plt.show()

        # Creating a CNL % Graph that represents each vintage
        await pyba.log("Creating Issuer/Vintage CNL % Graph...")
        cnl_issuer_df = df1[['seasoning_months', 'vintage', 'dealname', 'CUMMCHARGEOFFSAMT', 'orig_balance']]
        cnl_issuer_df = cnl_issuer_df.assign(
            IssuerYear=cnl_issuer_df.loc[:, 'dealname'].str.split().str[0].str.strip().astype(str) \
                       + ' ' + cnl_issuer_df.loc[:, 'vintage'].astype(str))

        cnl_issuer_df = cnl_issuer_df.reset_index(drop=True)
        cnl_issuer_df = (cnl_issuer_df.groupby(['seasoning_months', 'IssuerYear'])['CUMMCHARGEOFFSAMT'].sum() /
                         cnl_issuer_df.groupby(['seasoning_months', 'IssuerYear'])['orig_balance'].sum()) * 100
        cnl_issuer_df = cnl_issuer_df.unstack()

        fig, ax = plt.subplots()
        fig.suptitle('Cumulative Net Loss % By Vintage', fontsize=20)
        ax = cnl_issuer_df.plot(colormap='Set1', ax=ax, **pd_plot)
        plt.xlabel(**pd_xlabel)
        plt.ylabel('CNL %', fontsize=15)
        plt.legend(ncol=2, prop={'size': 14}, **pd_legend)
        plt.show()

    if displayGraphs == 'Yes' and sameIssuer == 'No' and compare == 'No':
        # Creating a Cumulative Net Loss Graph and a line representing the average
        await pyba.log("Creating CNL % Graph...")

        fig, ax = plt.subplots()
        fig.suptitle('Cumulative Net Loss By Deal', fontsize=20)
        cnl_df = df1.pivot_table(index='seasoning_months', columns='dealname', values='CUMCHARGEOFF')
        ax = cnl_df.plot(color=['r', 'g', 'b'], ax=ax, **pd_plot)
        plt.ylabel('CNL %', fontsize=15)
        plt.xlabel(**pd_xlabel)
        plt.legend(prop={'size': 12}, **pd_legend)
        plt.show()

        # Creating a Pool Factor vs CNL % Graph
        await pyba.log("Creating Pool Factor Graph...")
        pf_df = df1.pivot_table(index='seasoning_months', columns='dealname',
                                values=['POOLFACTOR', 'CUMCHARGEOFF', 'CUMGROSSCHARGEOFF'])
        pf_df['POOLFACTOR'] = pf_df['POOLFACTOR'] * 100
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        fig.suptitle('Pool Factor Vs. Cumulative Net Loss % & Cumulative Gross Loss % By Deal', fontsize=20)
        pf_df.plot(y='POOLFACTOR', legend=None, kind='bar', ax=ax,
                   color=['lightcoral', 'springgreen', 'lightsteelblue'], **pd_plot)
        pf_df.plot(y=['CUMCHARGEOFF'], secondary_y=True, legend=None, ax=ax2, color=['r', 'g', 'b'], **pd_plot)
        pf_df.plot(y=['CUMGROSSCHARGEOFF'], secondary_y=True, legend=None, ax=ax2, color=['r', 'g', 'b'],
                   linestyle='dashed', **pd_plot)
        plt.ylabel('CNL % % CGL %', fontsize=15)
        ax.set_ylabel('Pool Factor', fontsize=15)
        ax.set_xlabel(**pd_xlabel)
        plt.legend(prop={'size': 10}, **pd_legend)
        plt.show()

    if displayGraphs == 'Yes' and compare == 'Yes':
        await pyba.log("Creating Loss Severity % Graph...")
        comp_lossSev_df = df1[['seasoning_months', 'vintage', 'dealname', 'NETCHARGEOFFSAMT', 'GROSSCHARGEOFFSAMT']]
        comp_lossSev_df = comp_lossSev_df.assign(
            IssuerYear=comp_lossSev_df.loc[:, 'dealname'].str.split().str[0].str.strip().astype(str) \
                       + ' ' + comp_lossSev_df.loc[:, 'vintage'].astype(str))

        comp_lossSev_df = comp_lossSev_df.reset_index(drop=True)

        comp_lossSev_df = (comp_lossSev_df.groupby(['seasoning_months', 'IssuerYear'])['NETCHARGEOFFSAMT'].sum() /
                           comp_lossSev_df.groupby(['seasoning_months', 'IssuerYear'])[
                               'GROSSCHARGEOFFSAMT'].sum()) * 100
        comp_lossSev_df = comp_lossSev_df.unstack()

        fig, ax = plt.subplots()

        fig.suptitle('Loss Severity By Issuer / Vintage', fontsize=15)
        plt.xlabel('Deal Age (Months)', fontsize=15)
        plt.ylabel('Loss Severity %', fontsize=15)
        ax = comp_lossSev_df.plot(color=['r', 'r', 'r', 'r', 'g', 'g', 'g', 'g'], ax=ax, **pd_plot)
        ax.set_xlabel(**pd_xlabel)

        # Graph Formatting
        ax.lines[-4].set_marker("^"), ax.lines[-4].set_markerfacecolor("black"), ax.lines[-4].set_markeredgecolor(
            "black")
        ax.lines[-3].set_marker("*"), ax.lines[-3].set_markerfacecolor("black"), ax.lines[-3].set_markeredgecolor(
            "black")
        ax.lines[-2].set_marker("P"), ax.lines[-2].set_markerfacecolor("black"), ax.lines[-2].set_markeredgecolor(
            "black")
        ax.lines[-1].set_marker("D"), ax.lines[-1].set_markerfacecolor("black"), ax.lines[-1].set_markeredgecolor(
            "black")
        ax.lines[0].set_marker("^"), ax.lines[0].set_markerfacecolor("black"), ax.lines[0].set_markeredgecolor("black")
        ax.lines[1].set_marker("*"), ax.lines[1].set_markerfacecolor("black"), ax.lines[1].set_markeredgecolor("black")
        ax.lines[2].set_marker("P"), ax.lines[2].set_markerfacecolor("black"), ax.lines[2].set_markeredgecolor("black")
        ax.lines[3].set_marker("D"), ax.lines[3].set_markerfacecolor("black"), ax.lines[3].set_markeredgecolor("black")

        plt.legend(ncol=5, prop={'size': 12}, **pd_legend)
        plt.show()

        comp_cnl_issuer_df = df1[['seasoning_months', 'vintage', 'dealname', 'CUMMCHARGEOFFSAMT', 'orig_balance']]
        comp_cnl_issuer_df = comp_cnl_issuer_df.assign(
            IssuerYear=comp_cnl_issuer_df.loc[:, 'dealname'].str.split().str[0].str.strip().astype(str) \
                       + ' ' + comp_cnl_issuer_df.loc[:, 'vintage'].astype(str))
        comp_cnl_issuer_df = comp_cnl_issuer_df.reset_index(drop=True)
        comp_cnl_issuer_df = (comp_cnl_issuer_df.groupby(['seasoning_months', 'IssuerYear'])[
                                  'CUMMCHARGEOFFSAMT'].sum() /
                              comp_cnl_issuer_df.groupby(['seasoning_months', 'IssuerYear'])[
                                  'orig_balance'].sum()) * 100
        comp_cnl_issuer_df = comp_cnl_issuer_df.unstack()

        fig, ax = plt.subplots()
        fig.suptitle('Cumulative Net Loss % By Issuer/Vintage', fontsize=15)
        ax = comp_cnl_issuer_df.plot(color=['r', 'r', 'r', 'r', 'g', 'g', 'g', 'g'], ax=ax, **pd_plot)
        plt.xlabel('Deal Age (Months)', fontsize=15)
        plt.ylabel('CNL %', fontsize=15)

        ax.lines[-4].set_marker("^"), ax.lines[-4].set_markerfacecolor("black"), ax.lines[-4].set_markeredgecolor(
            "black")
        ax.lines[-3].set_marker("*"), ax.lines[-3].set_markerfacecolor("black"), ax.lines[-3].set_markeredgecolor(
            "black")
        ax.lines[-2].set_marker("P"), ax.lines[-2].set_markerfacecolor("black"), ax.lines[-2].set_markeredgecolor(
            "black")
        ax.lines[-1].set_marker("D"), ax.lines[-1].set_markerfacecolor("black"), ax.lines[-1].set_markeredgecolor(
            "black")
        ax.lines[0].set_marker("^"), ax.lines[0].set_markerfacecolor("black"), ax.lines[0].set_markeredgecolor("black")
        ax.lines[1].set_marker("*"), ax.lines[1].set_markerfacecolor("black"), ax.lines[1].set_markeredgecolor("black")
        ax.lines[2].set_marker("P"), ax.lines[2].set_markerfacecolor("black"), ax.lines[2].set_markeredgecolor("black")
        ax.lines[3].set_marker("D"), ax.lines[3].set_markerfacecolor("black"), ax.lines[3].set_markeredgecolor("black")

        plt.legend(ncol=2, prop={'size': 14}, **pd_legend)
        plt.show()

    df1 = df1.reindex(columns=fields).sort_values(by=['dealname', 'date'], kind='mergesort')
    df1 = df1.drop(columns=['closing_date'])
    fields.remove('closing_date')
    arr = df1.values

    await pyba.log("Retrieving Pool History...")
    for i in range(len(arr)):
        for y in range(len(fields)):
            await pyba.write_range(wsName, resStart, arr[i][y], offset_cols=y, offset_rows=x)
        x += 1
    await pyba.log("Done!")


'''
async def getKeyStats(wsName,portfolio,portfolioStart,vinStart,headersStart,inputStart,resStart):
    # do stuff here
    sfp = sfportal.SFPortal()
    await sfp.login('webbm', 'sav2015')

    i = 0
    x = 0
    pyba.log("Gathering Required Fields..")
    fields = [field for field in pyba.read_range(wsName, headersStart, columns=50) if len(field) > 0]

    pyba.log("Fields Gathered...")
    pyba.log("Gathering Requested CUSIPs..")
    cusips = [cusip for cusip in pyba.read_range_to_bottom(portfolio, portfolioStart)]

    pyba.log("CUSIPs Gathered...")
    desVintage = pyba.read_range(wsName, vinStart)
    pyba.log("Retrieving Key Stats...")

    for cusip in cusips:
        progDenom = len(pyba.read_range_to_bottom(portfolio, portfolioStart))
        sr = await sfportal.SearchCusipIsin.for_cusipisin(sfp,cusip)
        ks = await sfportal.KeyStats.for_search_results(sfp, [sr])

        pyba.write_range(wsName, resStart, cusip, offsetCols=-1, offsetRows=x)
        # This if statement is for cases where there is no data therefore it won't stop the code.
        if len(ks.results) == 0:
            pyba.write_range(wsName, resStart, "No Data", offsetCols=i, offsetRows=x)

        # This if statement is for cases where the deal doesn't belong to the desired Vintage
        elif int(ks.results[0]['$']['vintage']) < desVintage:

            pyba.write_range(wsName, resStart, "Older than desired vintage", offsetCols=i, offsetRows=x)

        else:
            for field in fields:
                if field in ks.results[0]['$'].keys():
                    res = ks.results[0]['$'][field]
                    pyba.write_range(wsName, resStart, res, offsetCols=i, offsetRows=x)
                    i += 1
                else:
                    pyba.write_range(wsName, resStart, "N/A", offsetCols=i, offsetRows=x)
                    i += 1

        i = 0
        #pyba.progress((x/progDenom)*100)
        x += 1
    #pyba.flush()
    pyba.log("Done!")

'''

if __name__ == '__main__':
    run(main)

# asyncio.run(main(pyba))
# pyba.done_excel()

'''
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except asyncio.CancelledError:
        pass
'''
