import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class KService {

  private apiUrl = 'http://localhost:5000/k';

  constructor(private http: HttpClient) { }

  trainK(): Observable<any> {
    const data = [
      { Feature1: 1.5, Feature2: 2.3 },
      { Feature1: 3.1, Feature2: 4.5 }
    ];
    const n_clusters = 3;

    return this.http.post(`${this.apiUrl}/run`, { data, n_clusters });
  }

  getImage(): Observable<any> {
    return this.http.get(`${this.apiUrl}/image`, { responseType: 'blob' });
  }

}
